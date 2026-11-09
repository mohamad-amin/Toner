"""Sharded Toner updates with asynchronous input statistics and factor computation."""

import concurrent.futures
import os
import time
import torch
import torch.distributed as dist

from . import muon as MU
from . import roots as PR
from .syrk import list_offsets, syrk_list_packed, unpack_ema_

KINDS_D = ("attn", "o", "mlp")  # hidden-width inputs: Q/K/V, attention output, MLP expansion
MATS = {"attn": ("q", "k", "v"), "o": ("o",), "mlp": ("gate", "up"), "down": ("down",)}


def _adamt(Ps, Gs, m, v, R2, n2, lr, wd, bc1, bc2, b1: float, b2: float, eps: float):
    """Update AdamT moments and return the parameter change for one matrix-shape group."""
    P, G = torch.stack(Ps), torch.stack(Gs).float()
    n = P.size(-1)
    m = m * b1 + G * (1 - b1)
    gw = PR.apply16_raw(G, R2)
    v = v * b2 + gw * gw * (1 - b2)
    U = PR.apply16_raw(m, R2) / bc1 / ((v / bc2).sqrt() + eps)
    UR = PR.apply16_raw(U, R2)
    upd = UR * (
        U.norm(dim=(-2, -1), keepdim=True) / UR.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    )
    dec = (
        PR.apply16_raw(PR.apply16_raw(P, R2), R2) * (n / n2)[:, None, None]
    )  # decay W R^2 / mean diag(R^2)
    return m, v, -(lr * wd) * dec - lr * upd


def _muont_in(Gs, m, R2, mu: float, nesterov: bool):
    """MuonT before Newton-Schulz: momentum, and (Nesterov) u R."""
    G = torch.stack(Gs).float()
    m = m * mu + G
    return m, PR.apply16_raw(G + mu * m if nesterov else m, R2)


def _muont_out(Ps, x, R2, n2, lr, wd, scale: float):
    """MuonT after Newton-Schulz: the parameter change -lr (wd W R^2 n / ||R||_F^2 + scale x R-hat), R-hat =
    R sqrt(n) / ||R||_F, as one product with R: -(lr wd n / ||R||_F^2 (W R) + lr scale sqrt(n) / ||R||_F x) R (three
    FP32-accurate R products per matrix with muont_in, instead of four)."""
    P = torch.stack(Ps)
    n = P.size(-1)
    V = ((lr * wd) * (n / n2))[:, None, None] * PR.apply16_raw(P, R2) + (
        lr * scale * n**0.5 / n2.sqrt().clamp_min(1e-30)
    )[:, None, None] * x
    return -PR.apply16_raw(V, R2)


def _muont_out_inner(Ps, x, lr, wd, scale: float):
    """Inner-root MuonT (--toner-outer 0): Muon's parameter change -lr (wd W + scale x) on X = NS(u R) (no R-hat and
    plain decoupled decay; toner_batch's "dir" arms)."""
    return -(lr * wd) * torch.stack(Ps) - (lr * scale) * x


adamt = torch.compile(_adamt, dynamic=False, options=PR._EXACT)
muont_in = torch.compile(_muont_in, dynamic=False, options=PR._EXACT)
muont_out = torch.compile(_muont_out, dynamic=False, options=PR._EXACT)
muont_out_inner = torch.compile(_muont_out_inner, dynamic=False, options=PR._EXACT)


def layer_counts(L, world, weights=None):
    """Layers owned by each rank (contiguous blocks): equal, or proportional to `weights` (largest remainder; a zero
    weight owns nothing). Owners do the roots, EMAs and updates of their layers' statistics and matrices, so a slower
    GPU can be given less of that work (every rank still computes its rows' partial Grams of every statistic)."""
    if not weights:
        assert L % world == 0, (
            "layers must split evenly over the ranks (or give --toner-owner-weights)"
        )
        return [L // world] * world
    w = [max(0.0, float(x)) for x in weights]
    assert len(w) == world and sum(w) > 0, (
        "--toner-owner-weights needs one non-negative weight per rank"
    )
    raw = [L * x / sum(w) for x in w]
    cnt = [int(x) for x in raw]
    for r in sorted(range(world), key=lambda q: raw[q] - cnt[q], reverse=True)[: L - sum(cnt)]:
        cnt[r] += 1
    return cnt


def offload_plan(world, weights, rows):
    """Row offload (--toner-offload): the ranks owning no layers (weight 0) send their rows of the offloaded statistic
    groups to the other ranks (receivers), which add them to their own Grams; an offloader then computes no Gram of
    those groups and is not in their reduce-scatter. Each offloader's rows [0, rows) are split into contiguous chunks
    over the receivers in proportion to their weights (largest remainder). Returns None (nothing to offload) or
    {"offloaders", "receivers", "chunks": {receiver: (lo, hi)}, "extra": {rank: rows received}}."""
    w = [max(0.0, float(x)) for x in weights] if weights else [1.0] * world
    off = [r for r in range(world) if w[r] == 0]
    recv = [r for r in range(world) if w[r] > 0]
    if not off or not recv:
        return None
    raw = [rows * w[r] / sum(w[q] for q in recv) for r in recv]
    cnt = [int(x) for x in raw]
    for i in sorted(range(len(recv)), key=lambda i: raw[i] - cnt[i], reverse=True)[
        : rows - sum(cnt)
    ]:
        cnt[i] += 1
    lo = [sum(cnt[:i]) for i in range(len(recv))]
    chunks = {r: (lo[i], lo[i] + cnt[i]) for i, r in enumerate(recv)}
    extra = {r: (len(off) * cnt[recv.index(r)] if r in recv else 0) for r in range(world)}
    return {"offloaders": off, "receivers": recv, "chunks": chunks, "extra": extra}


def gather_uneven(t, counts, group=None, async_op=False):
    """All-gather a rank's stack t (counts[rank], ...) into per-rank stacks (counts[r], ...): one NCCL all-gather into a
    tensor when the counts are equal, else the uneven list form. Returns (work or None, list of per-rank tensors)."""
    world = len(counts)
    if len(set(counts)) == 1:
        full = torch.empty((world,) + tuple(t.shape), device=t.device, dtype=t.dtype)
        work = dist.all_gather_into_tensor(full, t.contiguous(), group=group, async_op=async_op)
        return work, list(full.unbind(0))
    outs = [torch.empty((c,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype) for c in counts]
    work = dist.all_gather(outs, t.contiguous(), group=group, async_op=async_op)
    return work, outs


class StatGroup:
    """The statistics of one size, ordered by layer (so every rank's owned ones are one contiguous slice)."""

    def __init__(self, keys, n, bufs, counts, rank, device):
        self.keys, self.n, self.bufs = (
            keys,
            n,
            bufs,
        )  # keys: [(layer, kind)], bufs: rows buffers in key order
        k = len(keys)
        assert sum(counts) == k
        self.counts = counts  # statistics owned by each rank
        self.even = len(set(counts)) == 1
        self.per = counts[rank]
        lo = sum(counts[:rank])
        self.own = keys[lo : lo + self.per]
        self.offs = (
            None  # element offsets of the rows buffers (set at the first step, after placement)
        )
        m = (
            n * (n + 1) // 2
        )  # Grams are symmetric: packed upper triangles (half the reduce-scatter)
        self.partial = torch.empty(
            k, m, device=device
        )  # this rank's partial Grams of every statistic
        self.summed = torch.empty(
            self.per, m, device=device
        )  # the owned statistics' step sums (all ranks)
        self.C = torch.zeros(self.per, n, n, device=device)  # owned EMAs
        self.basis = (
            None  # roots.Basis of the owned statistics (None until the first exact decomposition)
        )
        self.basis_step = [-1] * self.per
        self.pending = [None] * self.per

        # worker threads make no CUDA calls but an event wait: theirs stalled the training thread's launches)
        self.host_in = torch.empty(self.per, n, n, pin_memory=True)
        self.host_out = torch.empty(self.per, n, n, pin_memory=True)
        self.host_tol = torch.empty(
            self.per, pin_memory=True
        )  # the refreshes' tolerances (floor-relative)
        self.tol = torch.full(
            (self.per,), 1e-4, device=device
        )  # each root's acceptance tolerance (graph input)
        self.stage = torch.empty(
            self.per, n, n, device=device
        )  # new bases on the GPU, installed in one call
        self.graph = None  # CUDA graph of the roots (captured after one eager step; static shapes and addresses)
        self.warm = False


class FastTonerCore:
    """AdamT or MuonT over the hidden matrices with exact every-step roots (see the module doc)."""

    def __init__(
        self,
        blocks,
        named,
        kind,
        *,
        hid,
        mlp,
        rows_total,
        args,
        world,
        rank,
        device,
        parameter_statistics=None,
    ):
        assert kind in ("adamt", "muont")
        self.kind, self.args, self.world, self.rank, self.device = kind, args, world, rank, device
        self.decay = args.toner_decay
        self.rows_total = rows_total
        self.iters = args.toner_iters
        # split inner / outer strengths (MuonT, train.py --toner-alpha-in / --toner-alpha-out): two roots per statistic,
        # cand / roots hold (R_in, R_out) pairs; otherwise exactly the single-root path
        self.split = bool(getattr(args, "toner_split", False))
        if self.split:
            assert kind == "muont" and getattr(args, "toner_root", "ref") == "ref"
            self.p_in, self.p_out = float(args.toner_alpha_in), float(args.toner_alpha_out)
            PR.set_powers(
                (self.p_in, self.p_out)
            )  # before any root (1/8, 1/4, 1/2; the iteration at 1/4)
        else:
            PR.set_alpha(args.toner_alpha)  # before any root (1/4 or 1/2)
        if getattr(args, "toner_root", "ref") == "fused":  # froot_state / froot_more
            from . import fused_root

            self.froot = fused_root
        else:
            self.froot = PR
        self.refresh = args.toner_refresh
        self.refresh_flagged = getattr(args, "toner_refresh_flagged", 1)
        L = len(blocks)
        self.L = L
        ow = getattr(args, "toner_owner_weights", "")
        self.layer_counts = layer_counts(
            L, world, [float(x) for x in ow.split(",")] if ow else None
        )
        self.owner_of_layer = [r for r, c in enumerate(self.layer_counts) for _ in range(c)]
        self.groups = {
            "d": StatGroup(
                [(i, k) for i in range(L) for k in KINDS_D],
                hid,
                [getattr(blocks[i], f"input_rows_{k}") for i in range(L) for k in KINDS_D],
                [len(KINDS_D) * c for c in self.layer_counts],
                rank,
                device,
            ),
            "f": StatGroup(
                [(i, "down") for i in range(L)],
                mlp,
                [blocks[i].input_rows_down for i in range(L)],
                list(self.layer_counts),
                rank,
                device,
            ),
        }
        if self.split:  # per-root tolerances (R_1/8, R_1/4, R_1/2; _tolerance, _split_over)
            for g in self.groups.values():
                g.host_tol = torch.empty(g.per, 3, pin_memory=True)
                g.tol = torch.full((g.per, 3), 1e-4, device=device)
        self.weight = 0.0
        # matrices: owned ones (by their layer's owner) grouped by shape; all of them, for the all-gather
        self.params, self.names, self.stat_of, self.name_of = [], [], {}, {}
        for n, p in named.items():
            if parameter_statistics is not None:
                if n not in parameter_statistics:
                    raise ValueError(f"No input-statistic binding for matrix {n}")
                i, kd = parameter_statistics[n]
                if p.ndim != 2 or not 0 <= i < L or kd not in (*KINDS_D, "down"):
                    raise ValueError(f"Invalid input-statistic binding for matrix {n}")
            elif n.startswith("body.blocks.") and p.ndim == 2:
                # Backward-compatible direct construction of the Llama core.
                _, _, i, sub, _ = n.split(".")
                i = int(i)
                kd = next(k for k, subs in MATS.items() if sub in subs)
            else:
                continue
            self.params.append(p)
            self.names.append(n)
            self.stat_of[p] = (i, kd)
            self.name_of[id(p)] = n
        self.param_groups = [
            {
                "params": self.params,
                "lr": 0.0,
                "lr_scale": args.toner_lr_mult,
                "weight_decay": args.wd,
            }
        ]
        by_shape = {}
        for p in self.params:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        self.shape_groups = []  # (shape, all params of the shape in a fixed order, per-rank owned lists)
        for shape, ps in by_shape.items():
            ps = sorted(ps, key=lambda q: (self.stat_of[q][0], self.name_of[id(q)]))
            per_rank = [
                [q for q in ps if self.owner_of_layer[self.stat_of[q][0]] == r]
                for r in range(world)
            ]
            self.shape_groups.append((shape, ps, per_rank))
        self.state = {}  # owned params' moments, stacked per shape group
        self.t = 0  # AdamT step count
        self.steps = 0
        self.fallbacks = 0
        self.extra = 0  # roots that needed more iterations (converged without a fallback)
        self.timing = {}
        self.check = {}  # validation (--toner-check-every): fast root vs the exact rule's FP32 eigh root
        self.gpu_ms = {}  # GPU time of the last completed step's phases
        self.main_stream = torch.cuda.current_stream(device)
        # statistics and roots: a side stream overlapping the last backward (priority -1: ahead of it, as step() waits
        # for it; or normal priority), or serial (the backward waits for them: no contention with its kernels)
        self.overlap = getattr(args, "toner_overlap", 1)
        self.side = torch.cuda.Stream(device=device, priority=-1 if self.overlap == 1 else 0)
        self.graph_pool = torch.cuda.graph_pool_handle()
        # Toner's collectives: their own communicator (concurrent with DDP's), or DDP's default one (in issue order)
        self.pg = (
            dist.new_group(backend="nccl")
            if world > 1 and getattr(args, "toner_own_pg", 1)
            else None
        )
        self.pg_stats = (
            self.pg
        )  # the statistics' reduce-scatter (--toner-stats-ctas: few SMs while it waits for peers)
        if world > 1 and getattr(args, "toner_stats_ctas", 0):
            o = dist.ProcessGroupNCCL.Options()
            o.config.max_ctas = args.toner_stats_ctas
            self.pg_stats = dist.new_group(backend="nccl", pg_options=o)
        # row offload (offload_plan): the groups whose rows the zero-weight ranks send to the others
        self.plan = (
            offload_plan(
                world, [float(x) for x in ow.split(",")] if ow else None, self.rows_total // world
            )
            if getattr(args, "toner_offload", "")
            else None
        )
        self.off_groups = set(args.toner_offload.split(",")) if self.plan else set()
        self.pg_recv = dist.new_group(self.plan["receivers"], backend="nccl") if self.plan else None
        self.is_offloader = bool(self.plan) and rank in self.plan["offloaders"]
        self.flags_host = torch.zeros(
            sum(g.per for g in self.groups.values()), dtype=torch.uint8, pin_memory=True
        )
        self.begun = False
        self.begin_timing = {}
        self.cand = None
        self._ev = self._ev_last = None
        self.chunks, self.ev_chunk = (
            0,
            None,
        )  # partial-Gram chunks so far this step (--toner-chunk); the last one's event
        workers = max(1, args.toner_host_workers)
        cores = (
            [int(c) for c in args.toner_host_cores.split(",")] if args.toner_host_cores else None
        )

        def init_worker():
            torch.cuda.set_device(device)
            if cores:
                os.sched_setaffinity(0, cores)

        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, initializer=init_worker
        )

    # ------------------------------------------------------------------ statistics (side stream)
    def chunk_stats(self):
        """--toner-chunk: right after the forward of a full chunk's last micro-batch (not the step's last), the chunk's
        partial Grams of every statistic on the side stream, added to the step's (FP32); the rows buffers then hold the
        next chunk. The step's last chunk is _stats()'s."""
        assert not self.off_groups, "--toner-chunk does not combine with --toner-offload"
        ev = torch.cuda.Event()
        ev.record(self.main_stream)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev)
            for g in self.groups.values():
                if g.offs is None:
                    g.offs = list_offsets(g.bufs)
                syrk_list_packed(g.bufs, g.offs, g.partial, acc=self.chunks > 0)
            self.ev_chunk = torch.cuda.Event()
            self.ev_chunk.record(self.side)
        self.chunks += 1

    def chunk_wait(self):
        """--toner-chunk: the main stream waits until the last chunk's Grams have read the rows buffers (call before the next
        chunk's first forward writes them)."""
        if self.ev_chunk is not None:
            self.main_stream.wait_event(self.ev_chunk)

    def _stats(self):
        """Partial Grams of every statistic (this rank's rows; with --toner-chunk, the step's last chunk added to the
        earlier ones), reduce-scatter to owners, EMA."""
        b = self.decay
        works = []
        for name, g in self.groups.items():
            off = name in self.off_groups
            if (
                off and self.is_offloader
            ):  # send this rank's rows of the group to the receivers; nothing else
                ops = [
                    dist.P2POp(dist.isend, buf[lo:hi], q, group=self.pg)
                    for buf in g.bufs
                    for q, (lo, hi) in self.plan["chunks"].items()
                ]
                works.extend(dist.batch_isend_irecv(ops))
                continue
            if (
                off
            ):  # receive the offloaders' rows into this rank's rows buffers (after its own rows)
                own = self.rows_total // self.world
                n = self.plan["chunks"][self.rank][1] - self.plan["chunks"][self.rank][0]
                ops = [
                    dist.P2POp(dist.irecv, buf[own + k * n : own + (k + 1) * n], o, group=self.pg)
                    for buf in g.bufs
                    for k, o in enumerate(self.plan["offloaders"])
                ]
                for w in dist.batch_isend_irecv(ops):
                    w.wait()
            if g.offs is None:
                g.offs = list_offsets(g.bufs)
            syrk_list_packed(g.bufs, g.offs, g.partial, acc=self.chunks > 0)
            pg, counts = (
                (self.pg_recv, [g.counts[r] for r in self.plan["receivers"]])
                if off
                else (self.pg_stats, g.counts)
            )
            if self.world == 1:
                g.summed.copy_(g.partial)
            elif len(set(counts)) == 1:
                works.append(
                    dist.reduce_scatter_tensor(g.summed, g.partial, group=pg, async_op=True)
                )
            else:  # weighted ownership: the uneven form (NCCL reduces per rank)
                works.append(
                    dist.reduce_scatter(
                        g.summed, list(g.partial.split(counts)), group=pg, async_op=True
                    )
                )
        for w in works:
            w.wait()
        for g in self.groups.values():
            if g.per:
                unpack_ema_(g.summed, g.C, b, (1 - b) / self.rows_total)
        self.weight = b * self.weight + (1 - b)
        self.chunks = 0

    # ------------------------------------------------------------------ bases (host)
    def _tolerance(self, lam):
        """Acceptance tolerance of roots from eigenvalues lam (..., n) of their statistic (--toner-tol). Split: (..., 3),
        those of R_1/8, R_1/4 and R_1/2, each from its own floor_base; the fraction --toner-tol-frac for R_1/2 when one
        is formed (0.2 / 3 by default, as alpha 1/2), 3x it for the others (0.2, as alpha 1/4); otherwise
        --toner-tol-frac for all."""
        if self.split:
            half = 0.5 in PR.POWERS
            cols = []
            for p in (0.125, 0.25, 0.5):
                if self.args.toner_tol == "fixed":
                    cols.append(
                        torch.full(
                            lam.shape[:-1],
                            self.args.toner_eps,
                            dtype=torch.float32,
                            device=lam.device,
                        )
                    )
                    continue
                frac = self.args.toner_tol_frac * (3.0 if half and p != 0.5 else 1.0)
                cols.append((frac * PR.floor_base_p(lam, p)).clamp_min(self.args.toner_tol_min))
            return torch.stack(cols, -1)
        if self.args.toner_tol == "fixed":
            return torch.full(
                lam.shape[:-1], self.args.toner_eps, dtype=torch.float32, device=lam.device
            )
        return (self.args.toner_tol_frac * PR.floor_base(lam)).clamp_min(self.args.toner_tol_min)

    def _host_refresh(self, x_in, x_out, tol_out, copied):
        """Worker thread: exact FP32 eigendecomposition of an owned statistic's snapshot (pinned x_in, once its copy
        is done) into pinned x_out, and its roots' tolerance into tol_out (0-d view). No other CUDA calls."""
        torch.set_num_threads(max(1, self.args.toner_host_threads))
        copied.synchronize()
        lam, V = torch.linalg.eigh(0.5 * (x_in + x_in.mT))
        x_out.copy_(V)
        tol_out.copy_(self._tolerance(lam))

    def _submit_refresh(self, g, j):
        """Side stream: snapshot of statistic j's EMA into pinned memory, then its exact host eigendecomposition."""
        g.host_in[j].copy_(g.C[j], non_blocking=True)
        copied = torch.cuda.Event(blocking=True)
        copied.record(self.side)
        g.pending[j] = (
            self.pool.submit(
                self._host_refresh, g.host_in[j], g.host_out[j], g.host_tol[j], copied
            ),
            self.steps,
        )

    def _install_ready(self):
        """Side stream: install the bases whose host decompositions have finished (one masked call per group)."""
        for g in self.groups.values():
            if not g.per:
                continue
            ready = [j for j in range(g.per) if g.pending[j] is not None and g.pending[j][0].done()]
            if not ready:
                continue
            mask = [False] * g.per
            for j in ready:
                g.pending[j][0].result()  # (re-raises a worker's exception)
                if (
                    g.pending[j][1] >= g.basis_step[j]
                ):  # not older than the basis in use (an exact fallback may
                    g.stage[j].copy_(
                        g.host_out[j], non_blocking=True
                    )  # have installed a newer one meanwhile)
                    g.tol[j].copy_(g.host_tol[j], non_blocking=True)
                    mask[j] = True
                    g.basis_step[j] = g.pending[j][1]
                g.pending[j] = None
            if any(mask):
                g.basis.install_many(
                    g.stage, torch.tensor(mask).pin_memory().to(self.device, non_blocking=True)
                )

    def _roots(self, g):
        """Root candidates of a group from its bases: R, B, the safeguard's flags (no host sync) and the iteration's
        state (flagged roots continue from it)."""
        a = self.args
        B, _ = PR.basis_transform(g.C, g.basis)
        lam, O, rho, est, state = self.froot.froot_state(B, self.iters, a.toner_freeze)
        if (
            self.split
        ):  # both roots from the same Y, each accepted at its own single-root path's strictness
            Rp, est_s = PR.form_powers(g.basis, lam, O)
            bad = (
                (rho > a.toner_rho_max)
                | self._split_over(est, est_s, g.tol)
                | ~torch.isfinite(lam).all(-1)
            )
            return (Rp[self.p_in], Rp[self.p_out]), B, bad, state
        bad = (rho > a.toner_rho_max) | ~(est <= g.tol) | ~torch.isfinite(lam).all(-1)
        return PR.form(g.basis, lam, O), B, bad, state

    def _split_over(self, est, est_s, tol):
        """Split roots above their tolerances (no host sync): tol (..., 3) holds each statistic's tolerances of R_1/8,
        R_1/4 and R_1/2 (_tolerance), est = est(Y). R_1/2: 2 est(Y), as the alpha-1/2 path; R_1/4: est(Y), as the
        alpha-1/4 path; R_1/8: est(Y) / 2 + est(sqrt)."""
        over = torch.zeros_like(est, dtype=torch.bool)
        if 0.5 in PR.POWERS:
            over |= ~(2 * est <= tol[..., 2])
        if 0.25 in PR.POWERS:
            over |= ~(est <= tol[..., 1])
        if 0.125 in PR.POWERS:
            over |= ~(0.5 * est + est_s <= tol[..., 0])
        return over

    def _roots_graphed(self, g):
        """_roots as a CUDA graph (one launch instead of ~40): eager once (compiles, autotunes), then captured on
        the side stream and replayed. Its inputs (EMA, bases) are updated in place, so their addresses never change;
        its outputs live in the graph's pool and are rewritten at every replay (used within the step)."""
        if not self.args.toner_graphs:
            return self._roots(g)
        if not g.warm:
            g.warm = True
            return self._roots(g)
        if g.graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=self.graph_pool, stream=self.side, capture_error_mode="thread_local"
            ):
                out = self._roots(g)
            g.graph = (graph,) + out
        g.graph[0].replay()
        return g.graph[1:]

    # ------------------------------------------------------------------ roots
    @torch.no_grad()
    def begin(self):
        """This step's statistics and root candidates on the side stream. Call right after the last micro-batch's
        forward (its rows are then complete), so they overlap its backward; step() finishes the step."""
        a = self.args
        tb0 = time.perf_counter()
        ev_fwd = torch.cuda.Event()
        ev_fwd.record(self.main_stream)
        self._ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.cand, flags = {}, []
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev_fwd)
            self._ev[0].record(self.side)
            if all(g.basis is not None for g in self.groups.values() if g.per):
                self._install_ready()
            tb1 = time.perf_counter()
            self._stats()
            self._ev[1].record(self.side)
            for name, g in self.groups.items():
                if not g.per:  # this rank owns none of the group's statistics (weighted ownership)
                    empty = torch.empty(0, g.n, g.n, device=self.device)
                    self.cand[name] = (
                        (empty, empty) if self.split else empty,
                        None,
                        torch.zeros(0, dtype=torch.bool, device=self.device),
                        None,
                    )
                elif (
                    g.basis is None or self.steps < a.toner_inline
                ):  # first steps / after a restart: exact roots
                    if self.split:
                        (V, Rp), lam_x = PR.exact_eigh_powers(g.C, True)
                        R = (Rp[self.p_in], Rp[self.p_out])
                    else:
                        (V, R), lam_x = PR.exact_eigh(
                            g.C, True
                        )  # (the statistic moves fastest at the start)
                    g.tol.copy_(self._tolerance(lam_x))
                    if g.basis is None:
                        g.basis = PR.Basis(V)
                    else:
                        for j in range(g.per):
                            g.basis.install(j, V[j])
                    g.basis_step = [self.steps] * g.per
                    g.pending = [None] * g.per
                    self.cand[name] = (
                        R,
                        None,
                        torch.zeros(g.per, dtype=torch.bool, device=self.device),
                        None,
                    )
                else:
                    self.cand[name] = self._roots_graphed(g)
                flags.append(self.cand[name][2])
            tb2 = time.perf_counter()
            self.flags_host.copy_(torch.cat(flags).to(torch.uint8), non_blocking=True)
            self.ev_flags = torch.cuda.Event()
            self.ev_flags.record(self.side)
            for g in self.groups.values():
                for j in range(g.per):
                    if (
                        g.pending[j] is None
                        and (self.steps + j) % self.refresh == 0
                        and g.basis_step[j] != self.steps
                    ):
                        self._submit_refresh(g, j)
            self._ev[2].record(self.side)
            self.ev_done = torch.cuda.Event()
            self.ev_done.record(self.side)
        for name, (
            R,
            B,
            _,
            _,
        ) in self.cand.items():  # used next on the main stream: keep their memory until those
            if self.groups[name].graph is None:  # uses finish (graph outputs are never freed)
                for t in R if self.split else (R,):
                    t.record_stream(self.main_stream)
                if B is not None:
                    B.record_stream(self.main_stream)
        if not self.overlap:  # serial: the backward waits for the statistics and roots
            self.main_stream.wait_stream(self.side)
        self.begun = True
        tb3 = time.perf_counter()
        self.begin_timing = {
            "install_host": tb1 - tb0,
            "stats_roots_host": tb2 - tb1,
            "snap_host": tb3 - tb2,
        }

    def _finish_roots(self):
        """Refine flagged factors on the side stream, then use exact fallback where needed."""
        a = self.args
        self.ev_flags.synchronize()  # the host waits for the side stream's flags only (not for the backward)
        flags = self.flags_host.tolist()
        roots, o, todo = {}, 0, []
        for name, g in self.groups.items():
            R, B, _, state = self.cand[name]
            bad = [j for j in range(g.per) if flags[o + j]]
            o += g.per
            roots[name] = R
            if bad:
                todo.append((g, R, B, state, bad))
        if todo:
            with torch.cuda.stream(self.side):
                for g, R, B, state, still in todo:
                    flagged = list(still)
                    cur, hopeless = None, []
                    for _ in range(a.toner_repair_rounds if state is not None else 0):
                        # converging but not yet within the tolerance: two more iterations for the flagged roots only,
                        # continuing from their state (the same iterates as froot with that many iterations), a few
                        # rounds; the subset padded to a power of two (few compiled shapes)
                        k = 1 << (len(still) - 1).bit_length()
                        pad = still + still[-1:] * (k - len(still))
                        idx = torch.tensor(pad, device=self.device)
                        if cur is None:
                            st = tuple(t.index_select(0, idx) for t in state)
                        else:  # the last round's state of the roots still flagged
                            pos = torch.tensor([cur[0].index(j) for j in pad], device=self.device)
                            st = tuple(t.index_select(0, pos) for t in cur[1])
                        lam, O, rho, est, dl = self.froot.froot_more(
                            B.index_select(0, idx), st, 2, a.toner_freeze
                        )
                        if self.split:
                            Rp2, est_s = PR.form_powers(
                                PR.Basis.of_splits(g.basis.S2.index_select(0, idx)), lam, O
                            )
                            R2 = (Rp2[self.p_in], Rp2[self.p_out])
                            okl = (
                                (rho <= a.toner_rho_max)
                                & ~self._split_over(est, est_s, g.tol.index_select(0, idx))
                                & torch.isfinite(lam).all(-1)
                            ).tolist()
                        else:
                            R2 = PR.form(
                                PR.Basis.of_splits(g.basis.S2.index_select(0, idx)), lam, O
                            )
                            okl = (
                                (rho <= a.toner_rho_max)
                                & (est <= g.tol.index_select(0, idx))
                                & torch.isfinite(lam).all(-1)
                            ).tolist()
                        rhok = (rho <= a.toner_rho_max).tolist()
                        for i, j in enumerate(still):
                            if okl[i]:
                                if self.split:
                                    R[0][j].copy_(R2[0][i])
                                    if R[1] is not R[0]:
                                        R[1][j].copy_(R2[1][i])
                                else:
                                    R[j].copy_(R2[i])
                        self.extra += sum(okl[: len(still)])
                        hopeless += [j for i, j in enumerate(still) if not okl[i] and not rhok[i]]
                        cur = (pad, (lam, O, dl))
                        still = [j for i, j in enumerate(still) if not okl[i] and rhok[i]]
                        if not still:
                            break
                    still = still + hopeless
                    for j in (
                        still
                    ):  # exact fallback: eigendecomposition in line, and the basis it gives (a host
                        self.fallbacks += (
                            1  # refresh still running for j is discarded when it lands: older)
                        )
                        if self.split:
                            (V, Rxp), lam_x = PR.exact_eigh_powers(g.C[j : j + 1], True)
                            g.tol[j : j + 1].copy_(self._tolerance(lam_x))
                            R[0][j].copy_(Rxp[self.p_in][0])
                            if R[1] is not R[0]:
                                R[1][j].copy_(Rxp[self.p_out][0])
                        else:
                            (V, Rx), lam_x = PR.exact_eigh(g.C[j : j + 1], True)
                            g.tol[j : j + 1].copy_(self._tolerance(lam_x))
                            R[j].copy_(Rx[0])
                        g.basis.install(j, V[0])
                        g.basis_step[j] = self.steps
                    if (
                        self.refresh_flagged
                    ):  # converged with more iterations from an old basis: refresh that basis
                        for j in (
                            flagged
                        ):  # now rather than at its turn (the next steps then need no extra work)
                            if j not in still and g.pending[j] is None:
                                self._submit_refresh(g, j)
            self.main_stream.wait_stream(self.side)
        else:
            self.main_stream.wait_event(self.ev_done)
        if (
            a.toner_check_every and self.steps % a.toner_check_every == 0
        ):  # validation only: an eigh per statistic
            for name, Rr in roots.items():
                g = self.groups[name]
                if not g.per:
                    continue
                if self.split:  # both roots: name (inner), name_out (outer)
                    _, Rxp = PR.exact_eigh_powers(g.C)
                    pairs = [(name, Rr[0], Rxp[self.p_in]), (name + "_out", Rr[1], Rxp[self.p_out])]
                else:
                    _, Rx = PR.exact_eigh(g.C)
                    pairs = [(name, Rr, Rx)]
                for key, R, Rx in pairs:
                    rel = (R - Rx).norm(dim=(-2, -1)) / Rx.norm(dim=(-2, -1))
                    Cn = (
                        g.C / g.C.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
                    )  # gradient weighting by C
                    d = R - Rx
                    ec = (
                        (
                            (d @ Cn @ d.mT).diagonal(dim1=-2, dim2=-1).sum(-1)
                            / (Rx @ Cn @ Rx.mT).diagonal(dim1=-2, dim2=-1).sum(-1)
                        )
                        .clamp_min(0)
                        .sqrt()
                    )
                    self.check[key] = (rel.max().item(), rel.median().item(), ec.max().item())
        return roots

    # ------------------------------------------------------------------ updates
    @torch.no_grad()
    def step(self):
        a = self.args
        g0 = self.param_groups[0]
        lr, wd = g0["lr"], g0["weight_decay"]
        if (
            self._ev_last is not None and self._ev_last[3].query()
        ):  # the last step's GPU phase times (ms)
            e = self._ev_last
            self.gpu_ms = {
                "stats": e[0].elapsed_time(e[1]),
                "roots": e[1].elapsed_time(e[2]),
                "update": e[2].elapsed_time(e[3]),
            }
        t0 = time.perf_counter()
        if not self.begun:  # begin() was not called after the last forward: do it here
            self.begin()
        self.begun = False
        roots = self._finish_roots()
        t1 = time.perf_counter()
        # per owned statistic: its R and ||R||_F^2 (R-hat's scale; mean diag R^2 = ||R||_F^2 / n); split: R_in, then
        # R_out and ||R_out||_F^2 (the outer factor's R-hat and decay)
        Rof = {}
        for name, g in self.groups.items():
            if self.split:
                Ri, Ro = roots[name]
                nrm2 = Ro.square().sum((-2, -1))
                for j, key in enumerate(g.own):
                    Rof[key] = (Ri[j], nrm2[j], Ro[j])
                continue
            R = roots[name]
            nrm2 = R.square().sum((-2, -1))
            for j, key in enumerate(g.own):
                Rof[key] = (R[j], nrm2[j])
        self.t += 1
        # step scalars on the device (no recompiles as they change, no sync)
        scal = torch.tensor([lr, wd, 1 - a.beta1**self.t, 1 - a.beta2**self.t]).pin_memory()
        lr_t, wd_t, bc1, bc2 = scal.to(self.device, non_blocking=True).unbind(0)
        works, deltas = [], []
        for shape, ps, per_rank in self.shape_groups:
            mine = per_rank[self.rank]
            if not mine:  # weighted ownership: nothing of this shape here (the all-gather still needs this rank)
                delta = torch.empty((0,) + tuple(shape), device=self.device)
            else:
                Rs = torch.stack([Rof[self.stat_of[q]][0] for q in mine])
                n2 = torch.stack([Rof[self.stat_of[q]][1] for q in mine])
                R2 = PR.rows_lo(
                    Rs
                )  # FP16 splits for the FP32-accurate products (the exact rule's FP32 products)
                st = self.state.setdefault(shape, {})
                grads = [q.grad for q in mine]
                if "m" not in st:
                    st["m"] = torch.zeros((len(mine),) + tuple(shape), device=self.device)
                    if self.kind == "adamt":
                        st["v"] = torch.zeros_like(st["m"])
                if self.kind == "muont":
                    st["m"], uR = muont_in(
                        grads, st["m"], R2, a.muon_momentum, bool(a.muon_nesterov)
                    )
                    x = MU.newton_schulz_batched(uR, a.muon_ns_steps, a.muon_eps, a.muon_ns_mode)
                    if self.split:  # the outer factor's root: X = NS(u R_in) R_out-hat, decay W R_out^2 / mean diag
                        R2o = PR.rows_lo(torch.stack([Rof[self.stat_of[q]][2] for q in mine]))
                        delta = muont_out(
                            mine, x, R2o, n2, lr_t, wd_t, max(1.0, shape[0] / shape[1]) ** 0.5
                        )
                    elif a.toner_outer:
                        delta = muont_out(
                            mine, x, R2, n2, lr_t, wd_t, max(1.0, shape[0] / shape[1]) ** 0.5
                        )
                    else:  # the inner root only (--toner-outer 0)
                        delta = muont_out_inner(
                            mine, x, lr_t, wd_t, max(1.0, shape[0] / shape[1]) ** 0.5
                        )
                else:
                    st["m"], st["v"], delta = adamt(
                        mine,
                        grads,
                        st["m"],
                        st["v"],
                        R2,
                        n2,
                        lr_t,
                        wd_t,
                        bc1,
                        bc2,
                        a.beta1,
                        a.beta2,
                        a.eps,
                    )
            deltas.append((ps, per_rank, delta))
            if self.world > 1:
                works.append(
                    gather_uneven(delta, [len(x) for x in per_rank], group=self.pg, async_op=True)
                )
            else:
                works.append((None, [delta]))
        for (work, gathered), (ps, per_rank, _) in zip(works, deltas):
            if work is not None:
                work.wait()
            for r in range(self.world):
                if per_rank[r]:
                    torch._foreach_add_(per_rank[r], list(gathered[r].unbind(0)))
        self._ev[3].record(self.main_stream)
        self._ev_last = self._ev
        self.steps += 1
        t2 = time.perf_counter()
        self.timing = {"roots_host": t1 - t0, "update_host": t2 - t1, **self.begin_timing}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return {}  # the real state is gathered collectively by full_state() (train.py's save)

    def load_state_dict(self, sd):
        pass

    @torch.no_grad()
    def full_state(self):
        """Collective checkpoint: gather statistic EMAs and parameter moments on rank zero."""
        self.main_stream.wait_stream(self.side)
        ema = {}
        for g in self.groups.values():
            full = torch.cat(gather_uneven(g.C, g.counts)[1]) if self.world > 1 else g.C
            if self.rank == 0:
                for (i, k), c in zip(g.keys, full.unbind(0)):
                    ema[f"{i}.{k}"] = c.cpu()
            del full
        moments = {}
        for shape, ps, per_rank in self.shape_groups:
            st = self.state.get(shape)
            keys = ("m",) if self.kind == "muont" else ("m", "v")
            for key in keys:
                if self.world == 1 and (st is None or key not in st):
                    continue  # (with several ranks a rank owning none of this shape still joins the gather)
                mine = (
                    st[key]
                    if st is not None and key in st
                    else torch.empty((0,) + tuple(shape), device=self.device)
                )
                parts = (
                    gather_uneven(mine, [len(x) for x in per_rank])[1] if self.world > 1 else [mine]
                )
                if self.rank == 0:
                    for r in range(self.world):
                        for q, v in zip(per_rank[r], parts[r].unbind(0)):
                            moments.setdefault(self.name_of[id(q)], {})[key] = v.cpu()
        return (
            {"ema": ema, "weight": self.weight, "moments": moments, "t": self.t}
            if self.rank == 0
            else None
        )

    @torch.no_grad()
    def snapshot(self, layers):
        """Collective: return bias-corrected statistics for selected layers on rank zero."""
        self.main_stream.wait_stream(self.side)
        out = {}
        for g in self.groups.values():
            full = torch.cat(gather_uneven(g.C, g.counts)[1]) if self.world > 1 else g.C
            if self.rank == 0:
                for (i, k), c in zip(g.keys, full.unbind(0)):
                    if i in layers:
                        out[f"{i}.{k}"] = (c / self.weight).cpu()
            del full
        return out if self.rank == 0 else None

    @torch.no_grad()
    def load_full_state(self, sd):
        """Every rank picks its owned statistics and moments by name (any world size)."""
        for g in self.groups.values():
            for j, (i, k) in enumerate(g.own):
                g.C[j].copy_(sd["ema"][f"{i}.{k}"])
        self.weight = sd["weight"]
        self.t = sd["t"]
        mom = sd[
            "moments"
        ]  # a resume without the moments would silently restart AdamT / MuonT: refuse it
        for shape, ps, per_rank in self.shape_groups:
            names = [self.name_of[id(q)] for q in per_rank[self.rank]]
            if not names:  # weighted ownership: none of this shape here
                continue
            missing = [nm for nm in names if nm not in mom]
            if missing:
                raise ValueError(f"checkpoint has no Toner moments for {missing[:3]}...")
            st = self.state.setdefault(shape, {})
            for key in mom[names[0]]:
                st[key] = torch.stack([mom[nm][key] for nm in names]).to(self.device)
