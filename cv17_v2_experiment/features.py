"""
features.py  --  Acoustic feature extraction for the "Human Voice is Unique"
replication experiment (Singh & Raj) on Mozilla Common Voice.

Implements the paper's canonical 41-feature list (Section 3), grouped by
production-chain stage.  Every feature is computed in its own guarded routine
that returns np.nan on failure -- features are NEVER imputed or fabricated.
Coverage (fraction of utterances for which a feature was successfully computed)
is reported downstream from the NaN pattern.

Audio convention: input is decoded to float32 mono and resampled to 16 kHz
before extraction.

Reliability tiers (documented in the report):
  A  well-established, low-risk          : F0, jitter, shimmer, CPP, dCPP,
                                           F1-F5, B1-B5, VTLE, spectral
                                           skewness/kurtosis/entropy/rolloff/
                                           flux, AlphaRatio, LHR, RMS, AMD,
                                           SemitoneSDF0, SpeechRate, BGD, (HNR aux)
  B  best-effort DSP, partial coverage   : SHR, IHI, GNE, SPI, SSPF, VFI, Nasality
  C  glottal inverse filtering / alignment: NAQ, CQ, GCT, SQ, MFDR (IAIF),
                                           VOT (requires forced alignment -> NOT MEASURED)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
import scipy.signal as sps
import scipy.stats as sstats
import parselmouth
from parselmouth.praat import call

SR = 16000
C_SOUND = 35000.0  # speed of sound cm/s (warm moist air)

# ---- canonical 41-feature list, in paper order -------------------------------
FEATURES_GLOTTAL = ["F0", "jitter", "shimmer", "GCT", "CQ", "MFDR", "SQ",
                    "NAQ", "SHR", "IHI", "VFI", "CPP"]
FEATURES_FILTER  = ["F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5",
                    "VTLE", "Nasality"]
FEATURES_SPECTRAL= ["SpectralSkewness", "SpectralKurtosis", "SpectralEntropy",
                    "SpectralRolloff", "SpectralFlux", "AlphaRatio", "LHR",
                    "SPI", "GNE", "dCPP"]
FEATURES_PROSODY = ["VOT", "SpeechRate", "BGD", "SemitoneSDF0", "AMD", "SSPF",
                    "RMS"]
FEATURES_41 = (FEATURES_GLOTTAL + FEATURES_FILTER + FEATURES_SPECTRAL +
               FEATURES_PROSODY)
assert len(FEATURES_41) == 41, len(FEATURES_41)
AUX = ["HNR"]  # extra, not part of the paper's 41 (used as cross-check only)

# ---- v2 CANONICAL 40-FEATURE LIST (VTLE REMOVED, regrouped per v2 brief) ------
# Identical SET to FEATURES_41 minus VTLE; grouping follows the v2 prompt.
# Internal feature keys (CamelCase) are kept for computation; DISPLAY maps to the
# v2 prompt's snake_case names for the report only.  VFP == VFI (vocal-fry index).
V2_GLOTTAL  = ["F0", "jitter", "shimmer", "GCT", "CQ", "MFDR", "SQ", "NAQ",
               "SHR", "IHI", "VFI", "SemitoneSDF0"]              # 12
V2_FILTER   = ["F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5",
               "Nasality"]                                       # 11  (VTLE removed)
V2_SPECTRAL = ["SpectralSkewness", "SpectralKurtosis", "SpectralEntropy",
               "SpectralRolloff", "SpectralFlux", "AlphaRatio", "LHR",
               "SPI", "GNE", "SSPF"]                             # 10
V2_PROSODY  = ["CPP", "dCPP", "RMS", "AMD", "SpeechRate", "VOT", "BGD"]  # 7
FEATURES_40 = V2_GLOTTAL + V2_FILTER + V2_SPECTRAL + V2_PROSODY
assert len(FEATURES_40) == 40, len(FEATURES_40)
assert set(FEATURES_40) == set(FEATURES_41) - {"VTLE"}, "v2 set must be 41 minus VTLE"

V2_GROUP = ({f: "glottal_source" for f in V2_GLOTTAL} |
            {f: "vocal_tract_filter" for f in V2_FILTER} |
            {f: "spectral_envelope" for f in V2_SPECTRAL} |
            {f: "articulatory_prosodic" for f in V2_PROSODY})

DISPLAY = {"SemitoneSDF0": "semitone_SD_F0", "VFI": "VFP",
           "SpectralSkewness": "spectral_skewness",
           "SpectralKurtosis": "spectral_kurtosis",
           "SpectralEntropy": "spectral_entropy",
           "SpectralRolloff": "spectral_rolloff",
           "SpectralFlux": "spectral_flux", "AlphaRatio": "alpha_ratio",
           "SpeechRate": "speech_rate"}
def disp(f): return DISPLAY.get(f, f)

DYNAMIC = {"jitter", "shimmer", "SpectralFlux", "dCPP"}  # summarized by mean


# ============================================================ helpers =========
def _nan(): return float("nan")

def _frame_sig(y, sr, win=0.04, hop=0.01):
    n = int(round(win * sr)); h = int(round(hop * sr))
    if len(y) < n:
        return None, None
    idx = range(0, len(y) - n + 1, h)
    frames = np.stack([y[i:i + n] for i in idx])
    win_f = np.hamming(n)
    return frames * win_f, n


# ============================================================ parselmouth ======
def praat_features(y, sr):
    """F0, jitter, shimmer, HNR, F1-F5, B1-B5, SemitoneSDF0, VTLE, F0 contour."""
    out = {k: _nan() for k in ["F0", "jitter", "shimmer", "HNR",
           "F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5",
           "VTLE", "SemitoneSDF0"]}
    f0_contour = None
    try:
        snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
    except Exception:
        return out, f0_contour
    # ---- pitch ----
    try:
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        vo = f0[f0 > 0]
        f0_contour = vo.copy()
        if vo.size >= 3:
            out["F0"] = float(np.mean(vo))
            semis = 12.0 * np.log2(vo / 55.0)  # ref 55 Hz (A1) -- SD is ref-invariant
            out["SemitoneSDF0"] = float(np.std(semis))
    except Exception:
        pass
    # ---- jitter / shimmer (need PointProcess) ----
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        out["jitter"] = float(call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        out["shimmer"] = float(call([snd, pp], "Get shimmer (local)",
                                    0, 0, 1e-4, 0.02, 1.3, 1.6))
    except Exception:
        pass
    # ---- HNR (aux) ----
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        h = call(harm, "Get mean", 0, 0)
        if np.isfinite(h) and h > -200:
            out["HNR"] = float(h)
    except Exception:
        pass
    # ---- formants F1-F5 & bandwidths B1-B5 ----
    try:
        form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                   maximum_formant=5500, window_length=0.025,
                                   pre_emphasis_from=50)
        ts = form.ts()
        Fv = {i: [] for i in range(1, 6)}
        Bv = {i: [] for i in range(1, 6)}
        for tt in ts:
            for i in range(1, 6):
                v = form.get_value_at_time(i, tt)
                b = call(form, "Get bandwidth at time", i, tt, "Hertz", "Linear")
                if v is not None and np.isfinite(v):
                    Fv[i].append(v)
                if b is not None and np.isfinite(b):
                    Bv[i].append(b)
        Fmeans = {}
        for i in range(1, 6):
            if len(Fv[i]) >= 3:
                out[f"F{i}"] = float(np.median(Fv[i])); Fmeans[i] = out[f"F{i}"]
            if len(Bv[i]) >= 3:
                out[f"B{i}"] = float(np.median(Bv[i]))
        # ---- VTLE: regress Fn on (2n-1) through origin -> slope = c/(4L) ----
        if len(Fmeans) >= 3:
            n = np.array([2 * i - 1 for i in Fmeans])
            f = np.array([Fmeans[i] for i in Fmeans])
            slope = np.sum(n * f) / np.sum(n * n)   # least squares through origin
            if slope > 0:
                L = C_SOUND / (4.0 * slope)         # cm
                if 8.0 < L < 25.0:                  # plausible human VTL
                    out["VTLE"] = float(L)
    except Exception:
        pass
    return out, f0_contour


# ============================================================ CPP / dCPP ======
def cpp_series(y, sr, win=0.04, hop=0.01, f0min=60, f0max=330):
    frames, n = _frame_sig(y, sr, win, hop)
    if frames is None:
        return np.array([])
    q = np.arange(n) / sr                       # quefrency (s)
    lo = int(np.floor(sr / f0max)); hi = int(np.ceil(sr / f0min))
    lo = max(lo, 2); hi = min(hi, n // 2)
    if hi <= lo + 2:
        return np.array([])
    vals = []
    eps = 1e-12
    qreg = q[lo:hi]
    for fr in frames:
        if np.sqrt(np.mean(fr ** 2)) < 1e-5:    # skip near-silent frames
            continue
        spec = np.fft.rfft(fr)
        logp = 10.0 * np.log10(np.abs(spec) ** 2 + eps)
        ceps = np.fft.irfft(logp)
        ceps = ceps[:n]
        seg = ceps[lo:hi]
        k = int(np.argmax(seg)); peak_q = qreg[k]; peak_v = seg[k]
        # regression line over the search range
        A = np.vstack([qreg, np.ones_like(qreg)]).T
        m, b = np.linalg.lstsq(A, seg, rcond=None)[0]
        line_v = m * peak_q + b
        vals.append(peak_v - line_v)
    return np.array(vals)


# ============================================================ spectral =========
def spectral_features(y, sr):
    out = {k: _nan() for k in ["SpectralSkewness", "SpectralKurtosis",
           "SpectralEntropy", "SpectralRolloff", "SpectralFlux",
           "AlphaRatio", "LHR", "SPI", "RMS", "AMD"]}
    try:
        n_fft = 1024; hop = 256
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2  # power
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        if S.shape[1] < 2:
            return out
        # per-frame magnitude distribution moments (mean over frames)
        mag = np.sqrt(S)
        colsum = mag.sum(axis=0) + 1e-12
        p = mag / colsum                          # prob over freq, per frame
        mean_f = (freqs[:, None] * p).sum(axis=0)
        var_f = ((freqs[:, None] - mean_f) ** 2 * p).sum(axis=0)
        sd_f = np.sqrt(var_f) + 1e-9
        skew = ((freqs[:, None] - mean_f) ** 3 * p).sum(axis=0) / sd_f ** 3
        kurt = ((freqs[:, None] - mean_f) ** 4 * p).sum(axis=0) / sd_f ** 4
        ent = -(p * np.log2(p + 1e-12)).sum(axis=0) / np.log2(p.shape[0])
        active = colsum > np.percentile(colsum, 20)   # ignore silent frames
        if active.sum() >= 3:
            out["SpectralSkewness"] = float(np.mean(skew[active]))
            out["SpectralKurtosis"] = float(np.mean(kurt[active]))
            out["SpectralEntropy"] = float(np.mean(ent[active]))
        # rolloff 95%
        ro = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.95)[0]
        if ro.size:
            out["SpectralRolloff"] = float(np.mean(ro[active] if active.sum() else ro))
        # spectral flux (frame-to-frame Euclidean change of normalised mag)
        magn = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12)
        flux = np.sqrt(((np.diff(magn, axis=1)) ** 2).sum(axis=0))
        if flux.size:
            out["SpectralFlux"] = float(np.mean(flux))
        # band energies
        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return S[m, :].sum() + 1e-12
        e_50_1k = band(50, 1000); e_1k_5k = band(1000, 5000)
        e_lt1k = band(0, 1000); e_gt3k = band(3000, sr / 2)
        out["AlphaRatio"] = float(10 * np.log10(e_1k_5k / e_50_1k))
        out["LHR"] = float(10 * np.log10(e_lt1k / e_gt3k))
        # SPI (MDVP): lower 70-1600 vs upper 1600-4500 harmonic energy ratio (dB)
        e_low = band(70, 1600); e_high = band(1600, 4500)
        out["SPI"] = float(10 * np.log10(e_low / e_high))
        # RMS (mean over frames, linear)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
        if rms.size:
            out["RMS"] = float(np.mean(rms))
        # AMD: amplitude-modulation depth of the <20 Hz envelope
        env = rms.astype(np.float64)
        env_sr = sr / hop
        if env.size > 10 and env.mean() > 0:
            # low-pass <20 Hz
            ny = env_sr / 2
            if ny > 20:
                b, a = sps.butter(2, 20.0 / ny, btype="low")
                env_lp = sps.filtfilt(b, a, env)
            else:
                env_lp = env
            m = env_lp.mean()
            if m > 0:
                out["AMD"] = float(np.std(env_lp) / m)   # coeff of variation
    except Exception:
        pass
    return out


# ============================================================ SHR / IHI ========
def shr_ihi(y, sr, f0_mean):
    """Subharmonic-to-harmonic ratio and inharmonicity index (best effort)."""
    shr = _nan(); ihi = _nan()
    if not (np.isfinite(f0_mean) and 60 < f0_mean < 500):
        return shr, ihi
    try:
        n_fft = 4096
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        spec = S.mean(axis=1)                      # avg magnitude spectrum
        spec = spec / (spec.max() + 1e-12)
        f0 = f0_mean
        def amp_at(f):
            if f >= freqs[-1]:
                return 0.0
            k = np.argmin(np.abs(freqs - f))
            lo = max(0, k - 2); hi = min(len(spec), k + 3)
            return float(spec[lo:hi].max())
        nharm = int(min(12, (sr / 2) / f0))
        harm_amps = [amp_at(h * f0) for h in range(1, nharm + 1)]
        sub_amps = [amp_at((h - 0.5) * f0) for h in range(1, nharm + 1)]
        H = np.sum(harm_amps); SUB = np.sum(sub_amps)
        if H > 0:
            shr = float(SUB / H)
        # IHI: deviation of measured harmonic peak from k*f0
        devs = []
        for h in range(2, nharm + 1):
            target = h * f0
            k = np.argmin(np.abs(freqs - target))
            band = slice(max(0, k - 8), min(len(spec), k + 9))
            if spec[band].size:
                pk = band.start + int(np.argmax(spec[band]))
                if spec[pk] > 0.02:
                    devs.append(abs(freqs[pk] - target) / target)
        if len(devs) >= 3:
            ihi = float(np.mean(devs))
    except Exception:
        pass
    return shr, ihi


# ============================================================ GNE ==============
def gne(y, sr):
    """Glottal-to-Noise Excitation ratio (Michaelis 1997), best effort."""
    try:
        # LPC inverse filter -> residual
        order = int(2 + sr / 1000)
        ye = librosa.effects.preemphasis(y)
        a = librosa.lpc(ye.astype(np.float64), order=order)
        res = sps.lfilter(a, [1.0], ye)
        # Hilbert envelopes in overlapping 1 kHz bands, hop 500 Hz
        centers = np.arange(1000, min(5000, sr / 2 - 500) + 1, 500)
        envs = []
        for c in centers:
            lo = (c - 500) / (sr / 2); hi = (c + 500) / (sr / 2)
            lo = max(lo, 1e-3); hi = min(hi, 0.999)
            if hi <= lo:
                continue
            b, aa = sps.butter(4, [lo, hi], btype="band")
            band = sps.lfilter(b, aa, res)
            env = np.abs(sps.hilbert(band))
            envs.append(env - env.mean())
        best = 0.0
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                if abs(centers[i] - centers[j]) >= 500:
                    a1 = envs[i]; a2 = envs[j]
                    d = np.sqrt((a1 @ a1) * (a2 @ a2))
                    if d > 0:
                        cc = float((a1 @ a2) / d)
                        best = max(best, cc)
        return best if best > 0 else _nan()
    except Exception:
        return _nan()


# ============================================================ SSPF / VFI =======
def sspf(y, sr):
    """Sibilant spectral peak frequency: peak of high-freq frication frames."""
    try:
        n_fft = 1024; hop = 256
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft,
                                                 hop_length=hop)[0]
        cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        energy = S.sum(axis=0)
        m = min(len(zcr), len(cent), S.shape[1])
        zcr, cent, energy, S = zcr[:m], cent[:m], energy[:m], S[:, :m]
        # frication: high ZCR, high centroid, sufficient energy
        if energy.size == 0:
            return _nan()
        thr_e = np.percentile(energy, 60)
        sel = (zcr > np.percentile(zcr, 80)) & (cent > 3000) & (energy > thr_e)
        if sel.sum() < 2:
            return _nan()
        hf = freqs >= 2000
        sub = S[hf][:, sel].mean(axis=1)
        if sub.size == 0 or sub.max() <= 0:
            return _nan()
        peak = freqs[hf][int(np.argmax(sub))]
        return float(peak)
    except Exception:
        return _nan()


def vfi(y, sr, f0_contour):
    """Vocal-fry index: fraction of voiced frames in creak regime (best effort)."""
    try:
        snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
        # low pitch-floor pass to catch creak (F0 ~ 25-80 Hz)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=30, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        vo = f0[f0 > 0]
        if vo.size < 5:
            return _nan()
        creak = np.mean(vo < 70.0)
        return float(creak)
    except Exception:
        return _nan()


def nasality(y, sr, f1):
    """Nasality index (best effort): low-frequency murmur (200-300 Hz) energy
    relative to F1 region energy.  Highly approximate; reported with coverage."""
    try:
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return S[m, :].sum() + 1e-12
        murmur = band(200, 300)
        f1c = f1 if (np.isfinite(f1) and f1 > 300) else 500.0
        oral = band(max(300, f1c - 150), f1c + 150)
        return float(10 * np.log10(murmur / oral))
    except Exception:
        return _nan()


# ============================================================ prosody ==========
def speechrate_bgd(y, sr):
    """Syllable-nucleus speech rate (De Jong style, approximate) and mean
    breath-group duration from silence segmentation."""
    sr_rate = _nan(); bgd = _nan()
    try:
        dur = len(y) / sr
        # intensity contour
        hop = int(0.01 * sr)
        rms = librosa.feature.rms(y=y, frame_length=int(0.03 * sr),
                                  hop_length=hop)[0]
        if rms.size < 5:
            return sr_rate, bgd
        intdb = 20 * np.log10(rms + 1e-9)
        peak = intdb.max()
        thr = peak - 25.0                              # 25 dB below peak
        # syllable nuclei = local maxima above thr, min 100 ms apart, dip>=2 dB
        pk, props = sps.find_peaks(intdb, height=thr, distance=int(0.10 / 0.01),
                                   prominence=2.0)
        if dur > 0.3:
            sr_rate = float(len(pk) / dur)             # syllables per second
        # breath groups: voiced/speech segments separated by silences > 0.30 s
        speech = intdb > (peak - 30.0)
        # find runs of speech
        seglens = []
        i = 0
        N = len(speech)
        sil_min = int(0.30 / 0.01)
        # collapse short silences
        run_start = None
        gap = 0
        for k in range(N):
            if speech[k]:
                if run_start is None:
                    run_start = k
                gap = 0
            else:
                gap += 1
                if run_start is not None and gap >= sil_min:
                    seglens.append((k - gap - run_start) * 0.01)
                    run_start = None
        if run_start is not None:
            seglens.append((N - run_start) * 0.01)
        seglens = [s for s in seglens if s > 0.05]
        if seglens:
            bgd = float(np.mean(seglens))
    except Exception:
        pass
    return sr_rate, bgd


# ============================================================ IAIF glottal =====
def iaif_glottal(y, sr, f0_mean):
    """Iterative Adaptive Inverse Filtering -> glottal flow & derivative.
    Returns NAQ, MFDR, CQ, GCT, SQ (best effort; expect partial coverage).

    Robust quantities (NAQ, MFDR) use peak flow / min-derivative / period.
    CQ/GCT/SQ require open/closed-phase estimation and are less reliable."""
    out = {k: _nan() for k in ["NAQ", "MFDR", "CQ", "GCT", "SQ"]}
    if not (np.isfinite(f0_mean) and 60 < f0_mean < 500):
        return out
    try:
        x = y.astype(np.float64)
        x = x - np.mean(x)
        # high-pass to remove low-freq drift
        b, a = sps.butter(2, 50.0 / (sr / 2), btype="high")
        x = sps.filtfilt(b, a, x)
        if np.max(np.abs(x)) < 1e-6:
            return out
        x = x / np.max(np.abs(x))
        # use a central voiced 0.5 s chunk (most stationary)
        if len(x) > int(0.6 * sr):
            mid = len(x) // 2; half = int(0.25 * sr)
            x = x[mid - half: mid + half]
        p_vt = int(2 + sr / 1000)
        # IAIF: 1) g1 = LPC order 1 ; inverse filter
        def lpc_inv(sig, order):
            a = librosa.lpc(sig, order=order)
            return sps.lfilter(a, [1.0], sig), a
        g1, _ = lpc_inv(x, 1)
        # 2) vocal tract estimate on g1
        v1, av = lpc_inv(g1, p_vt)
        # integrate residual of vocal-tract inverse filtering -> glottal flow
        # apply vocal tract inverse filter to x, then remove lip radiation (integrate)
        excitation = sps.lfilter(av, [1.0], x)
        # glottal flow = integrate excitation (cancel lip radiation 1-z^-1)
        flow = np.cumsum(excitation)
        flow = flow - sps.filtfilt(*sps.butter(2, 30.0 / (sr / 2), "high"), flow) * 0  # keep
        dflow = np.diff(flow, prepend=flow[0])
        T0 = sr / f0_mean                               # samples per period
        # segment into periods using flow-derivative negative peaks (GCIs ~ min dflow)
        neg = -dflow
        min_dist = int(0.7 * T0)
        gci, _ = sps.find_peaks(neg, distance=max(min_dist, 5),
                                height=np.percentile(neg, 75))
        if len(gci) < 4:
            return out
        naqs, mfdrs, cqs, sqs, gcts = [], [], [], [], []
        for i in range(len(gci) - 1):
            s, e = gci[i], gci[i + 1]
            per = e - s
            if not (0.5 * T0 < per < 2.0 * T0):
                continue
            seg_flow = flow[s:e] - flow[s:e].min()
            seg_d = dflow[s:e]
            fac = seg_flow.max()                         # peak-to-peak flow
            dmin = -seg_d.min()                          # max flow declination rate
            if fac <= 0 or dmin <= 0:
                continue
            naqs.append((fac / dmin) / per)
            mfdrs.append(dmin)
            # open/closed phase: flow above 50% of peak = open
            thr = 0.5 * fac
            openmask = seg_flow >= thr
            open_dur = openmask.sum()
            if open_dur > 0:
                cq = 1.0 - open_dur / per                # closed quotient approx
                cqs.append(np.clip(cq, 0, 1))
                gcts.append(np.clip(cq, 0, 1) * per / sr * 1000.0)  # ms
                # speed quotient: opening vs closing time within open phase
                peak_idx = int(np.argmax(seg_flow))
                op = peak_idx - np.where(openmask)[0][0]
                cl = np.where(openmask)[0][-1] - peak_idx
                if cl > 0 and op > 0:
                    sqs.append(op / cl)
        if len(naqs) >= 3:
            out["NAQ"] = float(np.median(naqs))
            out["MFDR"] = float(np.median(mfdrs))
        if len(cqs) >= 3:
            out["CQ"] = float(np.median(cqs))
            out["GCT"] = float(np.median(gcts))
        if len(sqs) >= 3:
            out["SQ"] = float(np.median(sqs))
    except Exception:
        pass
    return out


# ============================================================ main entry =======
def extract_features(y, sr):
    """Return dict of the 41 canonical features (+ HNR aux). NaN where not
    computable. VOT is intentionally NOT MEASURED (requires forced alignment)."""
    feats = {k: _nan() for k in FEATURES_41 + AUX}

    # resample to 16k mono
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if y.size < int(0.3 * sr) or np.max(np.abs(y)) < 1e-5:
        return feats  # too short / silent -> all NaN

    pr, f0_contour = praat_features(y, sr)
    feats.update({k: pr[k] for k in pr})

    cseries = cpp_series(y, sr)
    if cseries.size >= 3:
        feats["CPP"] = float(np.mean(cseries))
        feats["dCPP"] = float(np.mean(np.abs(np.diff(cseries))))

    feats.update(spectral_features(y, sr))

    sh, ih = shr_ihi(y, sr, feats["F0"])
    feats["SHR"] = sh; feats["IHI"] = ih
    feats["GNE"] = gne(y, sr)
    feats["SSPF"] = sspf(y, sr)
    feats["VFI"] = vfi(y, sr, f0_contour)
    feats["Nasality"] = nasality(y, sr, feats["F1"])

    sr_rate, bgd = speechrate_bgd(y, sr)
    feats["SpeechRate"] = sr_rate; feats["BGD"] = bgd

    feats.update(iaif_glottal(y, sr, feats["F0"]))

    # VOT: requires stop-burst detection + forced alignment -> NOT MEASURED
    feats["VOT"] = _nan()
    return feats
