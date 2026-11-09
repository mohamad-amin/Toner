"""Fast CUDA Toner optimizers. CUDA and Triton are imported lazily."""

from .adapter import FastAdamT, FastMuonT, ProjectionBlock, NanoGPTBlock

__all__ = ["FastMuonT", "FastAdamT", "ProjectionBlock", "NanoGPTBlock"]
