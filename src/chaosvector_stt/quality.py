"""Post-processing quality gates — silence, confidence, hallucination filtering."""

from __future__ import annotations

import re

import numpy as np

# Minimum RMS energy to run inference (skip silence)
MIN_RMS = 0.005

# Known STT hallucinations (model outputs these on silence/noise)
_HALLUCINATIONS = frozenset({
    "", "you", "thank you", "thanks", "bye", "goodbye",
    "the end", "subscribe", "like and subscribe",
    "i'm sorry", "thanks for watching",
    "please subscribe", "so", "okay",
})

# Family name corrections (word-boundary regex)
_NAME_CORRECTIONS: dict[re.Pattern, str] = {
    re.compile(r"\bJenny\b", re.IGNORECASE): "Jennie",
    re.compile(r"\bKinsley\b", re.IGNORECASE): "Kinzleigh",
    re.compile(r"\bKenzie\b", re.IGNORECASE): "Kinzleigh",
    re.compile(r"\bKinzley\b", re.IGNORECASE): "Kinzleigh",
    re.compile(r"\bLexy\b", re.IGNORECASE): "Lexi",
    re.compile(r"\bLexie\b", re.IGNORECASE): "Lexi",
    re.compile(r"\bZoe\b"): "Zoey",
    re.compile(r"\bEli\b"): "Eli",
    re.compile(r"\bclicks\b", re.IGNORECASE): "Plex",
    re.compile(r"\bClicks\b"): "Plex",
}


def is_silent(audio: np.ndarray) -> bool:
    """Check if audio is below minimum energy threshold."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < MIN_RMS


def is_hallucination(text: str) -> bool:
    """Check if transcription is a known hallucination."""
    return text.strip().lower().rstrip(".!?,") in _HALLUCINATIONS


def apply_name_corrections(text: str) -> str:
    """Apply family name corrections to STT output."""
    for pattern, replacement in _NAME_CORRECTIONS.items():
        text = pattern.sub(replacement, text)
    return text


def post_process(text: str) -> str:
    """Full post-processing pipeline: corrections + hallucination filter."""
    text = text.strip()
    if is_hallucination(text):
        return ""
    text = apply_name_corrections(text)
    return text
