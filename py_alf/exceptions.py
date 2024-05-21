"""Exceptions for pyALF"""

class TooFewBinsError(Exception):
    """Triggered when observable has too few bins for analysis."""

class AlreadyAnalyzed(Exception):
    """Triggered when parameters and bins are older than analysis results."""
