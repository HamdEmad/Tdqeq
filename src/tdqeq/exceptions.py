"""
Centralized exceptions for the Tdqeq pipeline.
"""


class TdqeqError(Exception):
    """Base exception for all Tdqeq errors."""

    pass


class PDFLoadError(TdqeqError):
    """Raised when PyMuPDF fails to load or process a PDF."""

    pass


class PDFCorruptError(PDFLoadError):
    """Raised when the PDF file data is corrupted."""

    pass


class PDFPasswordError(PDFLoadError):
    """Raised when the PDF requires a password."""

    pass


class DetectionError(TdqeqError):
    """Raised when table detection fails."""

    pass


class ExtractionError(TdqeqError):
    """Raised when table parsing/extraction fails."""

    pass


class ModelNotLoadedError(TdqeqError):
    """Raised when an AI model fails to load."""

    pass
