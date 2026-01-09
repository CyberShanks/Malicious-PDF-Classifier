"""
Trap4Phishing ML Pipeline
=========================
Modular pipeline for PDF phishing detection using structural features.
"""

from .features import PDFFeatureExtractor, extract_pdf_features

__version__ = "1.0.0"
__all__ = ["PDFFeatureExtractor", "extract_pdf_features"]
