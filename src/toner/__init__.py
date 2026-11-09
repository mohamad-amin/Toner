"""Toner: input-statistic preconditioning for Muon and AdamW."""

from .reference import MuonT, AdamT
from .statistics import ActivationStatistics

__all__ = ["MuonT", "AdamT", "ActivationStatistics"]
__version__ = "0.3.0"
