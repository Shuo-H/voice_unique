"""
mi_features.py -- lean per-utterance extractor for the information-theoretic
voice-individuality experiment.

It reuses ONLY the validated/low-risk acoustic routines from the project's
features.py (Praat via parselmouth, librosa/DSP spectral, CPP, syllable-rate
proxy).  It deliberately does NOT compute the glottal-source / inverse-filtering
family (GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE, Nasality, SSPF, VOT,
BGD): no validated glottal inverse-filtering tool (e.g. COVAREP / Aparat) is
available in this environment, so per the experiment's honesty rule those are
logged as NOT MEASURED with 0 coverage rather than fabricated from best-effort
DSP proxies.

MEASURED (28):
  Praat        : F0, F1..F5, B1..B5, jitter, shimmer, HNR
  librosa/DSP  : SpectralSkewness, SpectralKurtosis, SpectralEntropy,
                 SpectralRolloff, SpectralFlux, AlphaRatio, LHR, RMS, AMD,
                 CPP, dCPP
  formant      : VTLE
  prosodic     : SpeechRate (syllable-nucleus rate proxy), SemitoneSDF0

NOT MEASURED (14, no validated tool):
  GCT, CQ, NAQ, MFDR, SQ, SHR, IHI, VFI, SPI, GNE, Nasality, SSPF, VOT, BGD

Audio convention: decoded float mono, resampled to 16 kHz before extraction.
Features are NEVER imputed: a routine that fails yields np.nan and is counted
missing downstream.
"""
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa

# import the project's validated routines (features.py lives one dir up)
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import features as F  # noqa: E402

SR = 16000

MEASURED = [
    # Praat (parselmouth)
    "F0", "F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5",
    "jitter", "shimmer", "HNR",
    # librosa / DSP
    "SpectralSkewness", "SpectralKurtosis", "SpectralEntropy",
    "SpectralRolloff", "SpectralFlux", "AlphaRatio", "LHR", "RMS", "AMD",
    "CPP", "dCPP",
    # formant-based
    "VTLE",
    # prosodic
    "SpeechRate", "SemitoneSDF0",
]
assert len(MEASURED) == 28, len(MEASURED)

# glottal / inverse-filtering family -- intentionally NOT MEASURED (no validated tool)
NOT_MEASURED = ["GCT", "CQ", "NAQ", "MFDR", "SQ", "SHR", "IHI", "VFI", "SPI",
                "GNE", "Nasality", "SSPF", "VOT", "BGD"]

# dynamic (within-utterance variability) features kept per-utterance (not averaged away)
DYNAMIC = {"jitter", "shimmer", "SpectralFlux", "dCPP", "SemitoneSDF0", "AMD"}


def _nan():
    return float("nan")


def extract_measured(y, sr):
    """Return a dict of the 28 measured features for one utterance (np.nan on
    failure).  Mirrors features.extract_features' preprocessing but skips the
    glottal/inverse-filtering family entirely."""
    feats = {k: _nan() for k in MEASURED}

    # ---- preprocessing: mono, float32, 16 kHz ----
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if y.size < int(0.3 * sr) or np.max(np.abs(y)) < 1e-5:
        return feats  # too short / silent -> all NaN

    # ---- Praat: F0, jitter, shimmer, HNR, F1-5, B1-5, VTLE, SemitoneSDF0 ----
    pr, _ = F.praat_features(y, sr)
    for k in ["F0", "F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5",
              "jitter", "shimmer", "HNR", "VTLE", "SemitoneSDF0"]:
        feats[k] = pr.get(k, _nan())

    # ---- CPP / dCPP ----
    cseries = F.cpp_series(y, sr)
    if cseries.size >= 3:
        feats["CPP"] = float(np.mean(cseries))
        feats["dCPP"] = float(np.mean(np.abs(np.diff(cseries))))

    # ---- spectral / DSP (drop SPI: glottal group) ----
    sp = F.spectral_features(y, sr)
    for k in ["SpectralSkewness", "SpectralKurtosis", "SpectralEntropy",
              "SpectralRolloff", "SpectralFlux", "AlphaRatio", "LHR",
              "RMS", "AMD"]:
        feats[k] = sp.get(k, _nan())

    # ---- prosodic: syllable-nucleus rate proxy (BGD dropped: glottal group) ----
    sr_rate, _bgd = F.speechrate_bgd(y, sr)
    feats["SpeechRate"] = sr_rate

    return feats


if __name__ == "__main__":
    # smoke test on a synthetic vowel-ish signal
    rng = np.random.default_rng(1234)
    t = np.arange(int(1.5 * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * 130 * t)
         + 0.2 * np.sin(2 * np.pi * 260 * t)
         + 0.01 * rng.standard_normal(t.size)).astype(np.float32)
    d = extract_measured(y, SR)
    for k in MEASURED:
        print(f"{k:18s} {d[k]}")
