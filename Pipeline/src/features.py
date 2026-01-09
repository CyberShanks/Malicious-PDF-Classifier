"""
PDF Feature Extraction
Pulls out 40+ features from PDFs to detect phishing/malware.
Works for both batch processing and single file predictions.
"""

import os
import re
import math
import subprocess
import logging
from typing import Dict, Optional
import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)


class PDFFeatureExtractor:
    """Extracts features from PDFs - metadata, structure, content, and suspicious behavior."""

    def __init__(self, use_ocr: bool = True, ocr_max_pages: int = 1):
        """OCR is slow but needed for image-based PDFs. We only do the first page by default."""
        self.use_ocr = use_ocr
        self.ocr_max_pages = ocr_max_pages

    def extract(self, filepath: str) -> Dict[str, any]:
        """Pull all 40+ features from a PDF. Returns partial results if extraction fails."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        features = self._init_features(filepath)

        try:
            features["file_size"] = os.path.getsize(filepath)
            self._extract_pypdf2_features(filepath, features)
            self._extract_fitz_features(filepath, features)
        except Exception as e:
            logger.error(f"Feature extraction failed for {filepath}: {e}")
            # Still return what we got - partial features better than nothing

        return features

    def _init_features(self, filepath: str) -> Dict[str, any]:
        """Initialize all features with default values."""
        return {
            "file_path": filepath,
            "file_size": 0,
            "title_chars": 0,
            "encrypted": 0,
            "metadata_size": 0,
            "page_count": 0,
            "valid_pdf_header": 0,
            "image_count": 0,
            "text_length": 0,
            "object_count": 0,
            "font_object_count": 0,
            "embedded_file_count": 0,
            "average_embedded_file_size": 0,
            "stream_count": 0,
            "endstream_count": 0,
            "average_stream_size": 0,
            "entropy_of_streams": 0,
            "xref_count": 0,
            "xref_entries": 0,
            "name_obfuscations": 0,
            "total_filters": 0,
            "nested_filter_objects": 0,
            "objstm_count": 0,
            "js_count": 0,
            "javascript_count": 0,
            "uri_count": 0,
            "uses_nonstandard_port": 0,
            "action_count": 0,
            "aa_count": 0,
            "openaction_count": 0,
            "launch_count": 0,
            "submitform_count": 0,
            "acroform_count": 0,
            "xfa_count": 0,
            "jbig2decode_count": 0,
            "colors_count": 0,
            "richmedia_count": 0,
            "trailer_count": 0,
            "startxref_count": 0,
            "has_multiple_behavioral_keywords_in_one_object": 0,
            "used_ocr": 0
        }

    def _extract_pypdf2_features(self, filepath: str, features: Dict):
        """Uses PyPDF2 for metadata and keyword-based detection."""
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f, strict=False)

                features["encrypted"] = int(reader.is_encrypted)
                features["page_count"] = len(reader.pages)

                meta = reader.metadata
                if meta:
                    features["metadata_size"] = len(str(meta))
                    title = meta.get('/Title') or os.path.basename(filepath)
                    features["title_chars"] = len(str(title))

                f.seek(0)
                header = f.read(1024).decode(errors='ignore')
                features["valid_pdf_header"] = int(header.startswith('%PDF'))

                f.seek(0)
                raw = f.read().decode(errors='ignore')

                self._analyze_streams(raw, features)
                features["name_obfuscations"] = self._count_name_obfuscations(raw)
                self._extract_keyword_features(raw, features)
                self._detect_behavioral_overlap(raw, features)

        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {filepath}: {e}")

    def _extract_fitz_features(self, filepath: str, features: Dict):
        """PyMuPDF gives us structure info - images, fonts, objects, text."""
        try:
            doc = fitz.open(filepath)

            font_names = set()
            for page in doc:
                features["image_count"] += len(page.get_images(full=True))
                font_names.update([f[3] for f in page.get_fonts() if f[3]])

            features["font_object_count"] = len(font_names)
            features["object_count"] = doc.xref_length()
            features["xref_entries"] = sum(
                1 for i in range(doc.xref_length())
                if doc.xref_object(i, compressed=False)
            )
            features["text_length"] = sum(len(page.get_text()) for page in doc)

            doc.close()

        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {filepath}: {e}")
            self._fallback_text_extraction(filepath, features)

    def _analyze_streams(self, raw_content: str, features: Dict):
        """Streams hold binary data - we measure size and entropy."""
        features["stream_count"] = raw_content.count('stream')
        features["endstream_count"] = raw_content.count('endstream')

        matches = list(re.finditer(r'stream(.*?)endstream', raw_content, re.DOTALL))

        if matches:
            sizes = [len(m.group(1)) for m in matches if m.group(1)]
            features["average_stream_size"] = sum(sizes) / len(sizes) if sizes else 0

            entropies = [
                self._calculate_entropy(m.group(1))
                for m in matches if m.group(1)
            ]
            features["entropy_of_streams"] = sum(entropies) / len(entropies) if entropies else 0

    def _extract_keyword_features(self, raw_content: str, features: Dict):
        """Count suspicious keywords like /JS, /Launch, /URI etc."""
        keyword_map = {
            'objstm_count': '/ObjStm',
            'js_count': '/JS',
            'javascript_count': '/JavaScript',
            'uri_count': '/URI',
            'action_count': '/Action',
            'aa_count': '/AA',
            'openaction_count': '/OpenAction',
            'launch_count': '/Launch',
            'submitform_count': '/SubmitForm',
            'acroform_count': '/AcroForm',
            'xfa_count': '/XFA',
            'jbig2decode_count': '/JBig2Decode',
            'colors_count': '/Colors',
            'richmedia_count': '/RichMedia',
            'trailer_count': '/Trailer',
            'xref_count': '/Xref',
            'startxref_count': '/startxref',
            'total_filters': '/Filter',
            'nested_filter_objects': '/Filter ['
        }

        for feature_name, keyword in keyword_map.items():
            features[feature_name] = raw_content.count(keyword)

        # Weird ports like :8080 in URLs is sketchy
        if re.search(r'http[s]?://[^:\s]+:\d{4,5}', raw_content):
            features["uses_nonstandard_port"] = 1

    def _detect_behavioral_overlap(self, raw_content: str, features: Dict):
        """Multiple suspicious keywords in one object is extra sketchy."""
        objects = re.findall(r'obj(.*?)endobj', raw_content, re.DOTALL)
        behaviors = ['/JS', '/Launch', '/URI', '/OpenAction', '/SubmitForm', '/JavaScript', '/AA']

        count = 0
        for obj_content in objects:
            if sum(1 for keyword in behaviors if keyword in obj_content) >= 2:
                count += 1

        features["has_multiple_behavioral_keywords_in_one_object"] = count

    def _fallback_text_extraction(self, filepath: str, features: Dict):
        """Try pdfminer → pdftotext → OCR (in that order)."""
        text = self._extract_text_pdfminer(filepath)

        if not text.strip():
            text = self._extract_text_pdftotext(filepath)

        if not text.strip() and self.use_ocr:
            text = self._extract_text_ocr(filepath)
            features["used_ocr"] = 1

        features["text_length"] = len(text)

    @staticmethod
    def _extract_text_pdfminer(filepath: str) -> str:
        """Extract visible text using pdfminer."""
        try:
            return extract_text(filepath)
        except Exception:
            return ""

    @staticmethod
    def _extract_text_pdftotext(filepath: str) -> str:
        """Extract text using external pdftotext utility."""
        try:
            output = subprocess.check_output(
                ["pdftotext", filepath, "-"],
                stderr=subprocess.DEVNULL
            )
            return output.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _extract_text_ocr(self, filepath: str) -> str:
        """OCR fallback for image-based PDFs."""
        try:
            images = convert_from_path(
                filepath,
                dpi=200,
                first_page=1,
                last_page=self.ocr_max_pages
            )
            return "\n".join(pytesseract.image_to_string(img) for img in images)
        except Exception:
            return ""

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Compute Shannon entropy of text content."""
        if not text:
            return 0.0

        freq = [0] * 256
        for char in text:
            freq[ord(char) % 256] += 1

        freq = [p / len(text) for p in freq if p > 0]
        return -sum(p * math.log2(p) for p in freq)

    @staticmethod
    def _count_name_obfuscations(text: str) -> int:
        """Count obfuscated names (hex/escaped sequences)."""
        patterns = [
            r'/[a-zA-Z]*#\d{2}',
            r'/[a-zA-Z]*%[0-9a-fA-F]{2}',
            r'/[a-zA-Z]*\\x[0-9a-fA-F]{2}',
            r'/[a-zA-Z]*\\[0-7]{1,3}'
        ]
        return sum(len(re.findall(pattern, text)) for pattern in patterns)


# Convenience function for single-file extraction
def extract_pdf_features(filepath: str, use_ocr: bool = True) -> Dict[str, any]:
    """
    Convenience function to extract features from a single PDF.

    Args:
        filepath: Path to PDF file
        use_ocr: Whether to use OCR fallback

    Returns:
        Dictionary of extracted features
    """
    extractor = PDFFeatureExtractor(use_ocr=use_ocr)
    return extractor.extract(filepath)
