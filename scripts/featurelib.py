"""Per-utterance extraction of the 40 canonical features. Every feature is guarded:
on any failure it returns NaN so coverage is measured honestly (never imputed).

ALL 40 features are genuinely computed from the signal / PHN alignment. The
voice-source family (GCT, CQ, MFDR, SQ, NAQ) is derived from an approximate IAIF
glottal inverse filtering + GCI/threshold cycle segmentation. The remaining
non-standard names use documented operational definitions (see each function):
    SHR  - subharmonic/harmonic cepstral ratio
    IHI  - inter-harmonic intensity ratio (energy between harmonics vs at harmonics, dB)
    VFP  - vocal-fry probability (fraction of voiced frames flagged creaky)
    Nasality - low-frequency nasal-band energy ratio (proxy)
    SSPF - spectral-slope / spectral-tilt of the voiced log-spectrum (dB/kHz)
    BGD  - boundary-gap duration: mean pause/closure gap duration from PHN (s)
No value is imputed; coverage reflects how often each estimator yields a valid value.
"""
import warnings, math
import numpy as np

warnings.filterwarnings("ignore")

import parselmouth
from parselmouth.praat import call
from sphfile import SPHFile
import librosa
import scipy.signal
import scipy.stats

from common import (FEATURES_40, NONSPEECH_PHONES, STOP_RELEASES, VOWELS)

SR = 16000


def decode_wav(path):
    """Return (float64 signal in [-1,1], sr). Raises on empty/non-16k."""
    sph = SPHFile(path)
    sr = sph.format["sample_rate"]
    x = sph.content.astype(np.float64)
    if x.size == 0:
        raise ValueError("empty signal")
    if sr != SR:
        raise ValueError(f"unexpected sr={sr}")
    x = x / 32768.0
    return x, sr


def _safe(d, key, fn):
    try:
        v = fn()
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            d[key] = np.nan
        else:
            d[key] = float(v)
    except Exception:
        d[key] = np.nan


# ---------- glottal / pitch ----------

def pitch_feats(d, snd):
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    _safe(d, "F0", lambda: np.mean(voiced) if voiced.size else None)
    def semitone_sd():
        if voiced.size < 3:
            return None
        med = np.median(voiced)
        st = 12.0 * np.log2(voiced / med)
        return np.std(st, ddof=1)
    _safe(d, "semitone_SD_F0", semitone_sd)
    return pitch, voiced


def jitter_shimmer(d, snd):
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    except Exception:
        d["jitter"] = np.nan; d["shimmer"] = np.nan; return
    _safe(d, "jitter", lambda: call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
    _safe(d, "shimmer", lambda: call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))


def shr_feat(d, x, sr):
    """Subharmonic-to-harmonic ratio, cheap cepstral proxy: ratio of cepstral energy at
    2*T0 (subharmonic) to T0 (harmonic), averaged over voiced frames."""
    def fn():
        frame, hop = int(0.04 * sr), int(0.01 * sr)
        vals = []
        for i in range(0, len(x) - frame, hop):
            seg = x[i:i + frame] * np.hanning(frame)
            if np.sqrt(np.mean(seg**2)) < 1e-3:
                continue
            spec = np.abs(np.fft.rfft(seg)) + 1e-10
            cep = np.fft.irfft(np.log(spec))
            qmin, qmax = int(sr / 400), int(sr / 70)
            if qmax >= len(cep):
                continue
            region = cep[qmin:qmax]
            t0 = qmin + int(np.argmax(region))
            t_sub = 2 * t0
            if t_sub >= len(cep):
                continue
            h = abs(cep[t0]); s = abs(cep[t_sub])
            if h > 1e-9:
                vals.append(s / h)
        return np.mean(vals) if len(vals) > 5 else None
    _safe(d, "SHR", fn)


# ---------- vocal tract: formants + bandwidths ----------

def formant_feats(d, snd, pitch):
    try:
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                      maximum_formant=5500, window_length=0.025,
                                      pre_emphasis_from=50)
    except Exception:
        for i in range(1, 6):
            d[f"F{i}"] = np.nan; d[f"B{i}"] = np.nan
        return
    f0 = pitch.selected_array["frequency"]
    ts = pitch.ts()
    voiced_times = [t for t, v in zip(ts, f0) if v > 0]
    for i in range(1, 6):
        freqs, bws = [], []
        for t in voiced_times:
            fv = formant.get_value_at_time(i, t)
            bv = formant.get_bandwidth_at_time(i, t)
            if fv and not math.isnan(fv):
                freqs.append(fv)
            if bv and not math.isnan(bv):
                bws.append(bv)
        _safe(d, f"F{i}", lambda fr=freqs: np.mean(fr) if len(fr) > 3 else None)
        _safe(d, f"B{i}", lambda bb=bws: np.mean(bb) if len(bb) > 3 else None)


# ---------- spectral envelope ----------

def spectral_feats(d, x, sr):
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) + 1e-10
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = S**2
    framepow = power.sum(axis=0)
    active = framepow > (framepow.max() * 1e-4)
    Sa = S[:, active]; Pa = power[:, active]
    if Sa.shape[1] < 3:
        for k in ["spectral_skewness","spectral_kurtosis","spectral_entropy","spectral_rolloff",
                  "spectral_flux","alpha_ratio","LHR","SPI","GNE"]:
            d[k] = np.nan
        return

    # spectral moments (treat normalized magnitude over freq as a distribution)
    def moments():
        p = Sa / Sa.sum(axis=0, keepdims=True)
        mu = (freqs[:, None] * p).sum(axis=0)
        var = ((freqs[:, None] - mu) ** 2 * p).sum(axis=0)
        sd = np.sqrt(var) + 1e-10
        skew = (((freqs[:, None] - mu) ** 3) * p).sum(axis=0) / sd**3
        kurt = (((freqs[:, None] - mu) ** 4) * p).sum(axis=0) / sd**4
        return np.mean(skew), np.mean(kurt)
    try:
        sk, ku = moments()
    except Exception:
        sk, ku = None, None
    _safe(d, "spectral_skewness", lambda: sk)
    _safe(d, "spectral_kurtosis", lambda: ku)

    def entropy():
        p = Pa / Pa.sum(axis=0, keepdims=True)
        ent = -(p * np.log2(p + 1e-12)).sum(axis=0) / np.log2(p.shape[0])
        return np.mean(ent)
    _safe(d, "spectral_entropy", entropy)

    _safe(d, "spectral_rolloff",
          lambda: np.mean(librosa.feature.spectral_rolloff(S=Sa, sr=sr, roll_percent=0.85)))

    def flux():
        Sn = Sa / (Sa.sum(axis=0, keepdims=True) + 1e-10)
        diff = np.diff(Sn, axis=1)
        return np.mean(np.sqrt((diff**2).sum(axis=0)))
    _safe(d, "spectral_flux", flux)

    # alpha ratio: SPL(1-5kHz) - SPL(50-1000Hz) in dB
    def alpha():
        lo = (freqs >= 50) & (freqs < 1000)
        hi = (freqs >= 1000) & (freqs <= 5000)
        elo = Pa[lo, :].sum(); ehi = Pa[hi, :].sum()
        return 10 * np.log10((ehi + 1e-12) / (elo + 1e-12))
    _safe(d, "alpha_ratio", alpha)

    # LHR: low(<2kHz)/high(>2kHz) energy ratio in dB
    def lhr():
        lo = freqs < 2000; hi = freqs >= 2000
        return 10 * np.log10((Pa[lo, :].sum() + 1e-12) / (Pa[hi, :].sum() + 1e-12))
    _safe(d, "LHR", lhr)

    # SPI: soft phonation index = ratio low(70-1600) / high(1600-4500) harmonic energy, dB
    def spi():
        lo = (freqs >= 70) & (freqs < 1600); hi = (freqs >= 1600) & (freqs <= 4500)
        return 10 * np.log10((Pa[lo, :].sum() + 1e-12) / (Pa[hi, :].sum() + 1e-12))
    _safe(d, "SPI", spi)

    gne_feat(d, x, sr)


def gne_feat(d, x, sr):
    """Glottal-to-Noise Excitation ratio (Michaelis 1997, simplified): correlate Hilbert
    envelopes of adjacent 1-kHz bands of the LP inverse-filtered-ish signal; GNE = max corr."""
    def fn():
        sig = scipy.signal.lfilter([1, -0.95], 1, x)  # pre-emphasis as crude inverse filter
        bands = [(0, 1000), (500, 1500), (1000, 2000), (1500, 2500), (2000, 3000)]
        envs = []
        for lo, hi in bands:
            sos = scipy.signal.butter(4, [max(lo,1)/(sr/2), min(hi,sr/2-1)/(sr/2)], btype="band", output="sos")
            b = scipy.signal.sosfilt(sos, sig)
            env = np.abs(scipy.signal.hilbert(b))
            envs.append(env)
        cors = []
        for i in range(len(envs)):
            for j in range(i+1, len(envs)):
                a, b = envs[i] - envs[i].mean(), envs[j] - envs[j].mean()
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 1e-9:
                    cors.append(np.dot(a, b) / denom)
        return max(cors) if cors else None
    _safe(d, "GNE", fn)


# ---------- articulatory / prosodic ----------

def cpp_feats(d, x, sr):
    """CPP (mean) and dCPP (mean abs frame-to-frame change) over voiced-ish frames."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    cpps = []
    for i in range(0, len(x) - frame, hop):
        seg = x[i:i + frame] * np.hamming(frame)
        if np.sqrt(np.mean(seg**2)) < 1e-3:
            continue
        spec = np.abs(np.fft.rfft(seg)) + 1e-10
        logspec = 20 * np.log10(spec)
        cep = np.fft.irfft(logspec)
        qmin, qmax = int(sr / 400), int(sr / 60)
        if qmax >= len(cep):
            continue
        q = np.arange(len(cep))
        region = np.abs(cep[qmin:qmax])
        if region.size < 3:
            continue
        pk = qmin + int(np.argmax(region))
        # linear regression baseline over quefrency range
        qr = q[qmin:qmax]
        cr = np.abs(cep[qmin:qmax])
        A = np.vstack([qr, np.ones_like(qr)]).T
        slope, intercept = np.linalg.lstsq(A, cr, rcond=None)[0]
        baseline = slope * pk + intercept
        cpp = (np.abs(cep[pk]) - baseline)
        cpps.append(cpp)
    _safe(d, "CPP", lambda: np.mean(cpps) if len(cpps) > 3 else None)
    _safe(d, "dCPP", lambda: np.mean(np.abs(np.diff(cpps))) if len(cpps) > 4 else None)


def energy_feats(d, x, sr):
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    _safe(d, "RMS", lambda: np.mean(rms))
    # AMD: amplitude-modulation depth of the RMS envelope = std/mean (coeff of variation)
    def amd():
        r = rms[rms > rms.max() * 0.05]
        if r.size < 5 or r.mean() < 1e-9:
            return None
        return np.std(r) / np.mean(r)
    _safe(d, "AMD", amd)


def parse_phn(path):
    segs = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                segs.append((int(p[0]), int(p[1]), p[2]))
    return segs


def phn_feats(d, phn_path, sr=SR):
    try:
        segs = parse_phn(phn_path)
    except Exception:
        d["speech_rate"] = np.nan; d["VOT"] = np.nan; return
    # speech rate: speech phones / speech duration (excluding silence segments)
    def srate():
        sp = [(a, b, p) for (a, b, p) in segs if p not in NONSPEECH_PHONES]
        if not sp:
            return None
        dur = sum(b - a for a, b, p in sp) / sr
        return len(sp) / dur if dur > 0 else None
    _safe(d, "speech_rate", srate)
    # VOT: mean duration (s) of stop-release segments (b d g k p t)
    def vot():
        durs = [(b - a) / sr for (a, b, p) in segs if p in STOP_RELEASES]
        return np.mean(durs) if durs else None
    _safe(d, "VOT", vot)


# ---------- voice-source: IAIF glottal inverse filtering ----------

def _lpc_coef(sig, order):
    return librosa.lpc(sig.astype(np.float64), order=order)


def iaif_glottal_flow(x, sr):
    """Approximate IAIF (Alku 1992): returns (glottal_flow, glottal_flow_derivative).
    Stage1 order-1 glottal estimate -> vocal-tract LPC -> inverse filter -> integrate."""
    w = x * np.hamming(len(x))
    # stage 1: remove rough glottal contribution (order 1)
    a_g1 = _lpc_coef(w, 1)
    r1 = scipy.signal.lfilter(a_g1, [1.0], x)
    # stage 2: vocal-tract estimate
    p_vt = int(2 + sr // 1000)
    a_vt = _lpc_coef(r1 * np.hamming(len(r1)), p_vt)
    # cancel vocal tract from original -> glottal flow derivative
    gfd = scipy.signal.lfilter(a_vt, [1.0], x)
    # integrate (cancel lip radiation) -> glottal flow
    gf = np.cumsum(gfd - gfd.mean())
    gf = gf - scipy.signal.medfilt(gf, kernel_size=min(len(gf) // 2 * 2 + 1, 201)) if len(gf) > 201 else gf - gf.mean()
    return gf, gfd


def glottal_feats(d, x, sr, f0_mean):
    """GCT, CQ, MFDR, SQ, NAQ from IAIF flow segmented by GCIs (max-declination) +
    25% amplitude-threshold open/closed phase."""
    keys = ["GCT", "CQ", "MFDR", "SQ", "NAQ"]
    try:
        if not (f0_mean and 50 < f0_mean < 500):
            raise ValueError("no usable F0")
        gf, gfd = iaif_glottal_flow(x, sr)
        T0_samp = sr / f0_mean
        # GCIs = negative peaks of derivative (main excitation)
        peaks, _ = scipy.signal.find_peaks(-gfd, distance=max(int(0.6 * T0_samp), 2),
                                           prominence=np.std(gfd) * 0.5)
        gct, cq, mfdr, sq, naq = [], [], [], [], []
        for a, b in zip(peaks[:-1], peaks[1:]):
            if (b - a) < 0.4 * T0_samp or (b - a) > 2.5 * T0_samp:
                continue
            flow = gf[a:b]; deriv = gfd[a:b]
            n = len(flow)
            if n < 6:
                continue
            fmin, fmax = flow.min(), flow.max()
            amp = fmax - fmin
            if amp <= 1e-9:
                continue
            T0 = n / sr
            level = fmin + 0.25 * amp
            open_mask = flow >= level
            n_open = int(open_mask.sum())
            n_closed = n - n_open
            d_peak = -deriv.min()           # max flow declination (per-sample)
            if d_peak <= 1e-9:
                continue
            cq.append(n_closed / n)
            gct.append(n_closed / sr)
            mfdr.append(d_peak * sr)         # per-second
            naq.append((amp / (d_peak * sr)) / T0)
            # speed quotient: opening (start->peak) vs closing (peak->end) within open phase
            idx = np.where(open_mask)[0]
            if idx.size > 2:
                pk = int(np.argmax(flow))
                op = pk - idx[0]; cl = idx[-1] - pk
                if cl > 0 and op > 0:
                    sq.append(op / cl)
        _safe(d, "GCT", lambda: np.mean(gct) if len(gct) > 3 else None)
        _safe(d, "CQ", lambda: np.mean(cq) if len(cq) > 3 else None)
        _safe(d, "MFDR", lambda: np.mean(mfdr) if len(mfdr) > 3 else None)
        _safe(d, "SQ", lambda: np.mean(sq) if len(sq) > 3 else None)
        _safe(d, "NAQ", lambda: np.mean(naq) if len(naq) > 3 else None)
    except Exception:
        for k in keys:
            d[k] = np.nan


# ---------- harmonic / nasality / tilt proxies ----------

def harmonic_proxies(d, x, sr, f0_mean):
    """IHI (inter-harmonic intensity, dB), VFP (vocal-fry prob), Nasality (nasal band ratio),
    SSPF (spectral slope dB/kHz)."""
    n_fft, hop = 1024, 160
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) + 1e-10
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    framepow = P.sum(axis=0)
    active = framepow > framepow.max() * 1e-3
    Pa = P[:, active]
    if Pa.shape[1] < 3:
        for k in ["IHI", "Nasality", "SSPF"]:
            d[k] = np.nan
    else:
        # Nasality: energy in low nasal-murmur band (200-450 Hz) vs voiced band (0-3500)
        def nasality():
            nb = (freqs >= 200) & (freqs <= 450)
            vb = (freqs >= 0) & (freqs <= 3500)
            return 10 * np.log10((Pa[nb, :].sum() + 1e-12) / (Pa[vb, :].sum() + 1e-12))
        _safe(d, "Nasality", nasality)
        # SSPF: spectral slope of mean log-power vs freq (dB per kHz)
        def slope():
            mlog = 10 * np.log10(Pa.mean(axis=1) + 1e-12)
            A = np.vstack([freqs / 1000.0, np.ones_like(freqs)]).T
            s, _ = np.linalg.lstsq(A, mlog, rcond=None)[0]
            return s
        _safe(d, "SSPF", slope)
        # IHI: energy at inter-harmonic midpoints vs at harmonics, dB
        def ihi():
            if not (f0_mean and 50 < f0_mean < 500):
                return None
            harm_e, inter_e = [], []
            kmax = int(min(5000, sr / 2 - f0_mean) / f0_mean)
            for k in range(1, max(kmax, 2)):
                fh = k * f0_mean
                fi = (k + 0.5) * f0_mean
                ih = int(round(fh / (sr / n_fft)))
                ii = int(round(fi / (sr / n_fft)))
                if ii < Pa.shape[0]:
                    harm_e.append(Pa[ih, :].mean()); inter_e.append(Pa[ii, :].mean())
            if len(harm_e) < 2:
                return None
            return 10 * np.log10((np.mean(inter_e) + 1e-12) / (np.mean(harm_e) + 1e-12))
        _safe(d, "IHI", ihi)

    # VFP: fraction of voiced frames flagged creaky (very low F0)
    def vfp():
        pitch = parselmouth.Sound(x, sampling_frequency=sr).to_pitch(time_step=0.01,
                                                                     pitch_floor=50, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        if voiced.size < 5:
            return None
        return float(np.mean(voiced < 90.0))
    _safe(d, "VFP", vfp)


def bgd_feat(d, phn_path, sr=SR):
    """BGD = boundary-gap duration: mean duration (s) of pause/closure gap segments
    (pau, epi, and stop closures) within the utterance."""
    try:
        segs = parse_phn(phn_path)
    except Exception:
        d["BGD"] = np.nan; return
    gap_phones = {"pau", "epi", "bcl", "dcl", "gcl", "kcl", "pcl", "tcl"}
    durs = [(b - a) / sr for (a, b, p) in segs if p in gap_phones]
    _safe(d, "BGD", lambda: np.mean(durs) if durs else None)


# ---------- top-level ----------

def extract_utt(wav_path, phn_path):
    d = {k: np.nan for k in FEATURES_40}
    x, sr = decode_wav(wav_path)           # may raise -> caller logs decode failure
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch, _ = pitch_feats(d, snd)
    jitter_shimmer(d, snd)
    shr_feat(d, x, sr)
    formant_feats(d, snd, pitch)
    spectral_feats(d, x, sr)
    cpp_feats(d, x, sr)
    energy_feats(d, x, sr)
    phn_feats(d, phn_path, sr)
    glottal_feats(d, x, sr, d.get("F0"))
    harmonic_proxies(d, x, sr, d.get("F0"))
    bgd_feat(d, phn_path, sr)
    return d, len(x) / sr
