"""Feature extraction library for the TIMIT 40-feature battery.

Measured (attempted) features and the DSP/definition used are documented per
function. Features that cannot be extracted reliably without specialised
hardware (EGG), inverse-filtering of unknown reliability, or that have an
ambiguous definition are NOT attempted and are emitted as NaN:
    GCT, CQ, MFDR, SQ, NAQ, IHI, VFP  (glottal source, need EGG/inverse filter)
    Nasality                          (needs nasal accelerometer / nasometer)
    SSPF                              (ambiguous definition)
    BGD                              (ambiguous definition)
These are reported with 0 coverage and excluded -- never imputed.
"""
import numpy as np
import scipy.signal as sps
import scipy.fftpack as fftpack
from sphfile import SPHFile
import parselmouth
from parselmouth.praat import call

SR = 16000

# The canonical 40 (order preserved for reporting)
FEATURES_40 = [
    # Glottal source (12)
    "F0", "jitter", "shimmer", "GCT", "CQ", "MFDR", "SQ", "NAQ", "SHR", "IHI",
    "VFP", "semitone_SD_F0",
    # Vocal-tract filter (11)
    "F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5", "Nasality",
    # Spectral envelope (10)
    "spectral_skewness", "spectral_kurtosis", "spectral_entropy",
    "spectral_rolloff", "spectral_flux", "alpha_ratio", "LHR", "SPI", "GNE",
    "SSPF",
    # Articulatory / prosodic (7)
    "CPP", "dCPP", "RMS", "AMD", "speech_rate", "VOT", "BGD",
]

# Features deliberately NOT attempted (emit NaN, 0 coverage, excluded)
NOT_ATTEMPTED = {
    "GCT", "CQ", "MFDR", "SQ", "NAQ", "IHI", "VFP",  # glottal: need EGG/inv-filter
    "Nasality",                                       # need nasometer
    "SSPF", "BGD",                                    # ambiguous definition
}

STOPS_VOICELESS = {"p", "t", "k"}


def _nan_dict():
    return {f: np.nan for f in FEATURES_40}


def read_wav(path):
    sph = SPHFile(path)
    x = sph.content.astype(np.float64)
    sr = sph.format["sample_rate"]
    return x, sr


# ---------- spectral helpers ----------
def long_term_spectrum(x, sr, nfft=1024, hop=256):
    f, t, S = sps.stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                       window="hann", boundary=None)
    P = (np.abs(S) ** 2)  # power, freq x frames
    return f, P


def spectral_moments(f, P):
    """Power-weighted skewness & kurtosis of the long-term-average spectrum."""
    pm = P.mean(axis=1)
    s = pm.sum()
    if s <= 0:
        return np.nan, np.nan
    p = pm / s
    mu = np.sum(f * p)
    var = np.sum(((f - mu) ** 2) * p)
    if var <= 0:
        return np.nan, np.nan
    sd = np.sqrt(var)
    skew = np.sum(((f - mu) ** 3) * p) / (sd ** 3)
    kurt = np.sum(((f - mu) ** 4) * p) / (sd ** 4)  # raw (Fisher: -3 elsewhere)
    return float(skew), float(kurt)


def spectral_entropy_mean(P):
    """Mean per-frame normalised Shannon spectral entropy."""
    col = P.sum(axis=0)
    keep = col > 0
    if not keep.any():
        return np.nan
    Pk = P[:, keep]
    Pn = Pk / Pk.sum(axis=0, keepdims=True)
    H = -np.sum(Pn * np.log(Pn + 1e-12), axis=0)
    return float(np.mean(H / np.log(P.shape[0])))


def band_energy(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(P[m, :].mean(axis=1).sum()) if m.any() else 0.0


def alpha_ratio(f, P):
    lo = band_energy(f, P, 50, 1000)
    hi = band_energy(f, P, 1000, 5000)
    if lo <= 0 or hi <= 0:
        return np.nan
    return float(10 * np.log10(hi / lo))


def lhr(f, P):
    lo = band_energy(f, P, 0, 4000)
    hi = band_energy(f, P, 4000, 8000)
    if lo <= 0 or hi <= 0:
        return np.nan
    return float(10 * np.log10(lo / hi))


def spi(f, P):
    lo = band_energy(f, P, 70, 1600)
    hi = band_energy(f, P, 1600, 4500)
    if lo <= 0 or hi <= 0:
        return np.nan
    return float(10 * np.log10(lo / hi))


def spectral_rolloff(f, P, roll=0.85):
    pm = P.mean(axis=1)
    c = np.cumsum(pm)
    if c[-1] <= 0:
        return np.nan
    idx = np.searchsorted(c, roll * c[-1])
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def spectral_flux(x, sr, nfft=1024, hop=256):
    f, t, S = sps.stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                       window="hann", boundary=None)
    M = np.abs(S)
    if M.shape[1] < 2:
        return np.nan
    d = np.diff(M, axis=1)
    d[d < 0] = 0.0
    return float(np.mean(np.sqrt(np.sum(d ** 2, axis=0))))


# ---------- cepstral CPP ----------
def cpp_frames(x, sr, f0min=60, f0max=330, frame=0.04, hop=0.01):
    n = int(frame * sr)
    h = int(hop * sr)
    win = np.hanning(n)
    qmin = 1.0 / f0max
    qmax = 1.0 / f0min
    q = np.arange(n) / sr
    band = (q >= qmin) & (q <= qmax)
    if band.sum() < 4:
        return np.array([])
    vals = []
    for st in range(0, len(x) - n, h):
        seg = x[st:st + n] * win
        if np.sqrt(np.mean(seg ** 2)) < 1e-6:
            continue
        sp = np.fft.rfft(seg, n)
        logsp = np.log(np.abs(sp) ** 2 + 1e-12)
        full = np.concatenate([logsp, logsp[-2:0:-1]])
        cep = np.real(np.fft.ifft(full))
        cb = cep[band]
        qb = q[band]
        A = np.vstack([qb, np.ones_like(qb)]).T
        coef, *_ = np.linalg.lstsq(A, cb, rcond=None)
        reg = A @ coef
        peak_idx = np.argmax(cb)
        cpp = cb[peak_idx] - reg[peak_idx]
        vals.append(cpp)
    return np.array(vals)


# ---------- GNE (Michaelis 1997, standard form) ----------
def gne(x, sr):
    try:
        # downsample to 10 kHz
        g = 10000
        xr = sps.resample_poly(x, g, sr)
        xr = xr - np.mean(xr)
        if np.sqrt(np.mean(xr ** 2)) < 1e-6:
            return np.nan
        # LPC inverse filter -> residual (order 13)
        order = 13
        a = _lpc(xr, order)
        if a is None:
            return np.nan
        res = sps.lfilter(a, [1.0], xr)
        # Hilbert envelopes in bands, bw=1000, centres step 500
        bw = 1000.0
        centres = np.arange(1000, 5000 - 1, 400)
        envs = {}
        nyq = g / 2.0
        for c in centres:
            lo = (c - bw / 2) / nyq
            hi = (c + bw / 2) / nyq
            if lo <= 0 or hi >= 1:
                continue
            b, aa = sps.butter(4, [lo, hi], btype="band")
            yb = sps.lfilter(b, aa, res)
            envs[c] = np.abs(sps.hilbert(yb))
        cs = sorted(envs.keys())
        best = 0.0
        for i in range(len(cs)):
            for j in range(i, len(cs)):
                if abs(cs[i] - cs[j]) < bw / 2:
                    continue
                e1 = envs[cs[i]] - envs[cs[i]].mean()
                e2 = envs[cs[j]] - envs[cs[j]].mean()
                d = np.sqrt(np.sum(e1 ** 2) * np.sum(e2 ** 2))
                if d > 0:
                    r = np.sum(e1 * e2) / d
                    best = max(best, r)
        return float(best)
    except Exception:
        return np.nan


def _lpc(x, order):
    x = np.asarray(x, float)
    r = np.correlate(x, x, "full")[len(x) - 1:]
    if r[0] == 0:
        return None
    r = r[:order + 1]
    try:
        # Levinson-Durbin
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]
        for i in range(1, order + 1):
            acc = r[i] + np.sum(a[1:i] * r[i - 1:0:-1])
            k = -acc / e
            a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
            e *= (1 - k * k)
            if e <= 0:
                break
        return a
    except Exception:
        return None


# ---------- SHR (subharmonic-to-harmonic ratio, spectrum based) ----------
def shr_frames(x, sr, f0_series, frame=0.04):
    n = int(frame * sr)
    win = np.hanning(n)
    h = int(0.01 * sr)
    vals = []
    times = np.arange(0, len(x) - n, h) / sr
    for k, st in enumerate(range(0, len(x) - n, h)):
        t = st / sr
        # nearest f0 estimate
        f0 = _nearest(f0_series, t)
        if not (f0 and np.isfinite(f0) and f0 > 0):
            continue
        seg = x[st:st + n] * win
        if np.sqrt(np.mean(seg ** 2)) < 1e-6:
            continue
        nfft = 4096
        sp = np.abs(np.fft.rfft(seg, nfft))
        freqs = np.fft.rfftfreq(nfft, 1 / sr)
        H = 0.0
        SH = 0.0
        for m in range(1, 8):
            H += _amp_at(sp, freqs, m * f0)
        for m in range(1, 8):
            SH += _amp_at(sp, freqs, (m - 0.5) * f0)
        if H > 0:
            vals.append(SH / H)
    return np.array(vals)


def _amp_at(sp, freqs, fhz):
    if fhz <= 0 or fhz >= freqs[-1]:
        return 0.0
    i = np.argmin(np.abs(freqs - fhz))
    lo = max(0, i - 2)
    hi = min(len(sp), i + 3)
    return float(sp[lo:hi].max())


def _nearest(series, t):
    if series is None or len(series[0]) == 0:
        return None
    times, vals = series
    i = np.argmin(np.abs(times - t))
    return vals[i]


# ---------- PHN-based ----------
def parse_phn(path):
    segs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                segs.append((int(p[0]), int(p[1]), p[2]))
    return segs


SIL = {"h#", "pau", "epi", "1", "2"}


def speech_rate(segs, sr=SR):
    if not segs:
        return np.nan
    phones = [s for s in segs if s[2] not in SIL]
    if not phones:
        return np.nan
    # span from first to last non-silence phone
    start = phones[0][0]
    end = phones[-1][1]
    dur = (end - start) / sr
    if dur <= 0:
        return np.nan
    return len(phones) / dur


def vot(segs, sr=SR):
    """Positive VOT proxy: mean duration of voiceless stop release phones."""
    durs = [(b - a) / sr for a, b, lab in segs if lab in STOPS_VOICELESS]
    if not durs:
        return np.nan
    return float(np.mean(durs))


# ---------- main per-utterance ----------
def extract_one(wav, phn):
    out = _nan_dict()
    diag = {}
    x, sr = read_wav(wav)
    diag["n_samples"] = len(x)
    diag["sr"] = sr
    if len(x) == 0:
        diag["decode_fail"] = 1
        return out, diag
    diag["decode_fail"] = 0
    if sr != SR:
        diag["sr_mismatch"] = 1
        return out, diag
    diag["sr_mismatch"] = 0

    xf = x.astype(np.float64)
    xf = xf - np.mean(xf)
    mx = np.max(np.abs(xf))
    if mx > 0:
        xn = xf / mx
    else:
        xn = xf

    snd = parselmouth.Sound(xf, sampling_frequency=sr)

    # --- pitch / F0 ---
    f0_series = None
    try:
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0v = pitch.selected_array["frequency"]
        tt = pitch.xs()
        voiced = f0v[f0v > 0]
        f0_series = (tt[f0v > 0], f0v[f0v > 0])
        if voiced.size > 0:
            out["F0"] = float(np.mean(voiced))
        if voiced.size >= 2:
            semis = 12 * np.log2(voiced / np.median(voiced))
            out["semitone_SD_F0"] = float(np.std(semis, ddof=1))
    except Exception:
        pass

    # --- jitter / shimmer ---
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        npts = call(pp, "Get number of points")
        if npts >= 3:
            out["jitter"] = float(call(pp, "Get jitter (local)", 0, 0,
                                       1e-4, 0.02, 1.3))
            out["shimmer"] = float(call([snd, pp], "Get shimmer (local)", 0, 0,
                                        1e-4, 0.02, 1.3, 1.6))
    except Exception:
        pass

    # --- formants & bandwidths ---
    try:
        fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                 maximum_formant=5500)
        ts = fm.ts()
        for k in range(1, 6):
            fr = []
            bw = []
            for t in ts:
                v = fm.get_value_at_time(k, t)
                b = call(fm, "Get bandwidth at time", k, t, "hertz", "linear")
                if v and np.isfinite(v):
                    fr.append(v)
                if b and np.isfinite(b):
                    bw.append(b)
            if fr:
                out[f"F{k}"] = float(np.mean(fr))
            if bw:
                out[f"B{k}"] = float(np.mean(bw))
    except Exception:
        pass

    # --- spectral envelope ---
    try:
        f, P = long_term_spectrum(xf, sr)
        sk, ku = spectral_moments(f, P)
        out["spectral_skewness"] = sk
        out["spectral_kurtosis"] = ku
        out["spectral_entropy"] = spectral_entropy_mean(P)
        out["spectral_rolloff"] = spectral_rolloff(f, P)
        out["alpha_ratio"] = alpha_ratio(f, P)
        out["LHR"] = lhr(f, P)
        out["SPI"] = spi(f, P)
    except Exception:
        pass
    try:
        out["spectral_flux"] = spectral_flux(xf, sr)
    except Exception:
        pass
    try:
        out["GNE"] = gne(xf, sr)
    except Exception:
        pass

    # --- SHR ---
    try:
        sh = shr_frames(xn, sr, f0_series)
        if sh.size > 0:
            out["SHR"] = float(np.median(sh))
    except Exception:
        pass

    # --- CPP / dCPP ---
    try:
        cv = cpp_frames(xf, sr)
        if cv.size > 0:
            out["CPP"] = float(np.mean(cv))
        if cv.size > 1:
            out["dCPP"] = float(np.mean(np.abs(np.diff(cv))))
    except Exception:
        pass

    # --- RMS / AMD ---
    try:
        n = int(0.025 * sr)
        h = int(0.010 * sr)
        env = np.array([np.sqrt(np.mean(xf[i:i + n] ** 2))
                        for i in range(0, len(xf) - n, h)])
        env = env[env > 0]
        if env.size > 0:
            out["RMS"] = float(np.mean(env))
        if env.size > 1 and np.mean(env) > 0:
            out["AMD"] = float(np.std(env) / np.mean(env))
    except Exception:
        pass

    # --- PHN ---
    if phn:
        try:
            segs = parse_phn(phn)
            out["speech_rate"] = speech_rate(segs)
            out["VOT"] = vot(segs)
        except Exception:
            pass

    return out, diag
