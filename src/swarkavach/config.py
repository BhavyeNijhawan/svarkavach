"""Paths, signal-processing constants and runtime settings.

One import gives any module the project layout and the DSP parameters, so no
file has to hard-code a path or guess a frame length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR.parent
ROOT = SRC_DIR.parent

DATA_DIR = Path(os.environ.get("SWARKAVACH_DATA", ROOT / "data"))
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_AUDIO_DIR = CORPUS_DIR / "audio"
CORPUS_CALLS_DIR = CORPUS_DIR / "calls"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
RAW_DIR = DATA_DIR / "raw"
UPLOADS_DIR = DATA_DIR / "uploads"

SERVER_DIR = ROOT / "server"
STATIC_DIR = SERVER_DIR / "static"
DOCS_DIR = ROOT / "docs"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to. Safe to call repeatedly."""
    for d in (
        DATA_DIR,
        CORPUS_DIR,
        CORPUS_AUDIO_DIR,
        CORPUS_CALLS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        RAW_DIR,
        UPLOADS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------

#: Everything in this project runs at telephone bandwidth. ASVspoof 2021
#: showed that models trained on 16 kHz studio audio fall apart once a real
#: codec touches the signal, so 8 kHz is the native rate here, not an
#: afterthought applied at test time.
TARGET_SR = 8000

#: Rate used when a module genuinely needs wideband audio (speaker embedding
#: front ends trained at 16 kHz, Whisper).
WIDEBAND_SR = 16000


@dataclass
class FrameConfig:
    """Short-time analysis settings (syllabus module 5)."""

    sr: int = TARGET_SR
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    window: str = "hamming"
    preemphasis: float = 0.97
    n_fft: int = 512
    center: bool = False

    @property
    def frame_len(self) -> int:
        return int(round(self.sr * self.frame_ms / 1000.0))

    @property
    def hop_len(self) -> int:
        return int(round(self.sr * self.hop_ms / 1000.0))


@dataclass
class CepstralConfig:
    """Filterbank and cepstral settings (syllabus module 6)."""

    n_filters: int = 40          # filterbank channels
    n_ceps: int = 20             # cepstral coefficients kept
    # The filterbank stops where the conditioning band-pass stops. With
    # fmin 60 and fmax 3800 the outer filters read only the filter's
    # roll-off, and on a set whose real half is 8 kHz telephone audio and
    # whose spoofs were 24 kHz renders, the energy above 3400 Hz alone
    # separated the classes at AUC 1.000 through exactly those filters.
    fmin: float = 300.0
    fmax: float = 2800.0         # same edges as antispoof.features.condition
    lifter: int = 22
    use_energy: bool = True
    deltas: bool = True
    delta_width: int = 2
    cmvn: bool = True


@dataclass
class ProsodyConfig:
    """Pitch and voice-quality settings, used by the PIM feature."""

    f0_min: float = 60.0
    f0_max: float = 400.0
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    voicing_threshold: float = 0.30


@dataclass
class VADConfig:
    """Energy plus zero-crossing voice activity detection."""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    energy_percentile: float = 35.0
    min_speech_ms: float = 120.0
    min_silence_ms: float = 160.0
    pad_ms: float = 40.0


@dataclass
class PipelineConfig:
    """Top-level switches for an end-to-end run."""

    alert_threshold: float = 0.65        # risk value that raises an alert
    asr_backend: str = "auto"            # auto | whisper | gold | manual
    antispoof_backend: str = "auto"      # auto | gmm | gbm | rawnet | heuristic
    intent_backend: str = "auto"         # auto | tfidf | bert | rules
    ner_backend: str = "auto"            # auto | crf | bilstm | rules
    fusion_model: str = "logreg"         # logreg | gbm
    stream_realtime_factor: float = 1.0  # 1.0 = wall-clock, >1 = faster replay
    seed: int = 20230100                 # matches the student registration number


@dataclass
class Settings:
    frame: FrameConfig = field(default_factory=FrameConfig)
    cepstral: CepstralConfig = field(default_factory=CepstralConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SETTINGS = Settings()

# --------------------------------------------------------------------------
# Corpus generation defaults
# --------------------------------------------------------------------------

DEFAULT_N_CALLS = 480
DEFAULT_SCAM_RATIO = 0.5
DEFAULT_SYNTHETIC_RATIO = 0.5
#: The fusion layer is a stacked model: it consumes the branch outputs, so
#: fitting it on rows the branches were themselves trained on means fitting
#: it to in-sample branch predictions that never occur at test time.
#: Measured, CRF entity F1 was 1.000 on those rows against 0.887 on test.
#: So branches train on `train`, the fusion trains on `dev`, and `dev` is
#: sized large enough to fit 25 features rather than left as a token slice.
DEFAULT_SPLIT = {"train": 0.50, "dev": 0.25, "test": 0.25}

#: Speaker-disjoint splitting needs a speaker pool big enough that no speaker
#: leaks across a boundary. 24 gives roughly 20 calls per speaker at the
#: default corpus size.
N_SPEAKERS = 24

# --------------------------------------------------------------------------
# Project identity, used in reports and the dashboard header
# --------------------------------------------------------------------------

PROJECT_NAME = "SwarKavach"
PROJECT_TAGLINE = "Voice-clone fraud call detection for Hinglish telephony"
PROJECT_VERSION = "1.0.0"
COURSE_CODE = "BCSE419L"


def artifact_path(name: str) -> Path:
    """Where a trained model artifact lives."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / name


def result_path(name: str) -> Path:
    """Where an evaluation result JSON lives."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def find_ffmpeg() -> Optional[str]:
    """Return the ffmpeg binary path if the system has one, else None.

    Codec simulation degrades gracefully: with ffmpeg we get real GSM-FR and
    AMR-NB, without it we still get exact G.711 companding implemented in
    numpy, which is a real telephony codec in its own right.
    """
    import shutil

    return shutil.which("ffmpeg")
