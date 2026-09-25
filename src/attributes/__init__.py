"""TBPS canonical attributes and query-specific discrimination."""

from .attribute_scoring import AttributeGallery, SUPPORTED_SLOTS
from .text_extractor import TextAttributeExtractor, extract, extract_with_provenance

__all__ = [
    "AttributeGallery", "SUPPORTED_SLOTS",
    "TextAttributeExtractor", "extract", "extract_with_provenance",
]
