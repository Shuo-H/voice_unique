#!/usr/bin/env python3
"""
Voice-uniqueness experiment -- STEP 1: feature extraction over TIMIT.

Honesty policy: every feature is wrapped in try/except and returns np.nan on
failure. Nothing is imputed/interpolated/fabricated. Features that are not
implementable from a single audio channel (Nasality) or have no reliable
single-session estimator implemented (VFI) are emitted as np.nan BY DESIGN and
will show 0 coverage -> marked NOT MEASURED downstream.

Provenance of each feature is recorded in FEATURE_SOURCE below.
"""
import os, sys, glob, json, warnings, math
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SEED = 1234
np.random.seed(SEED)

SR = 16000
# Override with env VU_TIMIT_ROOT (point at the dir containing TRAIN/ and TEST/).
TIMIT_ROOT = os.environ.get("VU_TIMIT_ROOT",
                            "/sessions/lucid-sleepy-shannon/data/timit/TIMIT")
# Output dir: env VU_OUT, else the parent of this script's dir (deliverable layout:
# scripts/ lives under the results folder), else the current working directory.
def _resolve_out():
    env = os.environ.get("VU_OUT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.abspath(os.path.join(here, ".."))
    return parent if os.path.basename(here).lower() == "scripts" else os.getcwd()
OUT_DIR = _resolve_out()

# ----------------------------------------------------------------------------
# Canonical feature set (operationalizes the paper's ~41 features; HNR retained
# explicitly as an auxiliary glottal-source measure -> 42 candidate columns).
# ----------------------------------------------------------------------------
FEATURE_ORDER = [
    # glottal source
    "F0","jitter","shimmer","HNR","CPP","dCPP","NAQ","CQ","GCT","MFDR","SQ",
    "SHR","IHI","VFI","SPI","GNE",
    # vocal-tract filter
    "F1","F2","F3","F4","F5","B1","B2","B3","B4","B5","VTLE","Nasality",
    # spectral envelope
    "spectral_skewness","spectral_kurtosis","spectral_entropy","spectral_rolloff",
    "spectral_flux","alpha_ratio","LHR","RMS","AMD",
    # articulatory / prosodic
    "VOT","speech_rate","BGD","semitone_SD_F0","SSPF",
]

FEATURE_CATEGORY = {f: "glottal_source" for f in
    ["F0","jitter","shimmer","HNR","CPP","dCPP","NAQ","CQ","GCT","MFDR","SQ","SHR","IHI","VFI","SPI","GNE"]}
FEATURE_CATEGORY.update({f: "vocal_tract_filter" for f in
    ["F1","F2","F3","F4","F5","B1","B2","B3","B4","B5","VTLE","Nasality"]})
FEATURE_CATEGORY.update({f: "spectral_envelope" for f in
    ["spectral_skewness","spectral_kurtosis","spectral_entropy","spectral_rolloff","spectral_flux","alpha_ratio","LHR","RMS","AMD"]})
FEATURE_CATEGORY.update({f: "articulatory_prosodic" for f in
    ["VOT","speech_rate","BGD","semitone_SD_F0","SSPF"]})

# Features deliberately not measured (documented; emitted as NaN, never invented)
NOT_MEASURED_BY_DESIGN = {
    "Nasality": "requires nasalance (oral+nasal channels); not estimable from single TIMIT channel",
    "VFI": "no reliable single-session vocal-fry detector implemented; would be fabrication to guess",
}

# TIMIT phone classes
VOWELS = {"iy","ih","eh","ey","ae","aa","aw","ay","ah","ao","oy","ow","uh","uw",
          "ux","er","ax","ix","axr","ax-h"}
SYLLABIC = {"el","em","en","eng"}            # syllabic consonants count as nuclei
NUCLEI = VOWELS | SYLLABIC
VLESS_STOP_REL = {"p","t","k"}               # release segments (VOT proxy)
SILENCE = {"h#","pau","epi"}
SIBILANTS = {"s","sh","z","zh"}

C_SOUND = 35000.0  # speed of sound, cm/s (for VTLE)

# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_audio(path):
    """Decode NIST SPHERE TIMIT .WAV. Verify non-empty, 16 kHz. Raise on failure."""
    from sphfile import SPHFile
    try:
        sph = SPHFile(path)
        sr = int(sph.format["sample_rate"])
        x = sph.content.astype(np.float64)
    except Exception:
        import soundfile as sf
        x, sr = sf.read(path)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x[:, 0]
    if x.size == 0:
        raise ValueError(f"empty audio: {path}")
    if sr != SR:
        raise ValueError(f"unexpected sample rate {sr} for {path}")
    m = np.max(np.abs(x))
    if m > 0:
        x = x / 32768.0 if m > 1.5 else x
    return x, sr

def read_phn(path):
    seg = []
    if not os.path.exists(path):
        return seg
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 3:
                seg.append((int(p[0]), int(p[1]), p[2]))
    return seg

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _safe(d, key, fn):
    try:
        v = fn()
        if v is None: return
        v = float(v)
        if np.isfinite(v):
            d[key] = v
    except Exception:
        return

# ----------------------------------------------------------------------------
# main per-file extractor
# ----------------------------------------------------------------------------
def extract_one(path):
    import parselmouth
    from parselmouth.praat import call
    import scipy.signal as ss
    import scipy.stats as sstat
    import librosa

    out = {f: np.nan for f in FEATURE_ORDER}
    # meta
    spk = os.path.basename(os.path.dirname(path))
    utt = os.path.splitext(os.path.basename(path))[0]
    sex = "M" if spk[0].upper() == "M" else "F"
    out["_speaker_id"] = spk
    out["_sex"] = sex
    out["_utt_id"] = utt
    out["_path"] = path

    try:
        x, sr = read_audio(path)
    except Exception as e:
        out["_error"] = f"read:{e}"
        return out
    out["_dur_s"] = len(x) / sr

    # sex-specific pitch range
    if sex == "M":
        f0min, f0max, fmax = 60.0, 320.0, 5000.0
    else:
        f0min, f0max, fmax = 90.0, 500.0, 5500.0

    snd = parselmouth.Sound(values=x, sampling_frequency=sr)

    # ---- pitch ----
    f0v = np.array([])
    pitch_t = pitch_f0 = None
    try:
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0min, pitch_ceiling=f0max)
        pitch_f0 = pitch.selected_array["frequency"]
        pitch_t = pitch.xs()
        f0v = pitch_f0[pitch_f0 > 0]
        if f0v.size:
            out["F0"] = float(np.mean(f0v))
            out["semitone_SD_F0"] = float(np.std(12.0 * np.log2(f0v / 1.0)))
    except Exception:
        pass
    out["_voiced_frac"] = float(f0v.size / max(1, (len(pitch_f0) if pitch_f0 is not None else 1)))

    # ---- jitter / shimmer / HNR via PointProcess ----
    pp = None
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
        _safe(out, "jitter", lambda: call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        _safe(out, "shimmer", lambda: call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
    except Exception:
        pass
    try:
        harm = snd.to_harmonicity_cc(0.01, f0min, 0.1, 1.0)
        hv = harm.values[harm.values != -200]
        if hv.size:
            out["HNR"] = float(np.mean(hv))
    except Exception:
        pass

    # ---- formants F1-F5 + bandwidths B1-B5 (sampled at voiced frames) ----
    try:
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                      maximum_formant=fmax, window_length=0.025,
                                      pre_emphasis_from=50.0)
        if pitch_t is not None and f0v.size:
            vt = pitch_t[pitch_f0 > 0]
        else:
            vt = formant.xs()
        Fs = {n: [] for n in range(1, 6)}
        Bs = {n: [] for n in range(1, 6)}
        for t in vt:
            for n in range(1, 6):
                fv = formant.get_value_at_time(n, t)
                bv = formant.get_bandwidth_at_time(n, t)
                if fv is not None and np.isfinite(fv): Fs[n].append(fv)
                if bv is not None and np.isfinite(bv): Bs[n].append(bv)
        meanF = {}
        for n in range(1, 6):
            if Fs[n]:
                meanF[n] = float(np.nanmean(Fs[n])); out[f"F{n}"] = meanF[n]
            if Bs[n]:
                out[f"B{n}"] = float(np.nanmean(Bs[n]))
        # VTLE from uniform closed-open tube: L_n=(2n-1)c/(4 F_n); avg F1..F4
        Ls = [(2 * n - 1) * C_SOUND / (4.0 * meanF[n]) for n in range(1, 5) if n in meanF and meanF[n] > 0]
        if Ls:
            out["VTLE"] = float(np.mean(Ls))
    except Exception:
        pass

    # ---- STFT-based spectral block (shared) ----
    n_fft, hop, win = 1024, 160, 400
    try:
        S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window="hann"))
        P = S ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # len n_fft/2+1
        stft_t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop, n_fft=n_fft)
        # voiced mask aligned to stft frames via nearest pitch frame
        vmask = np.zeros(S.shape[1], dtype=bool)
        if pitch_t is not None and pitch_f0 is not None and len(pitch_t):
            idx = np.searchsorted(pitch_t, stft_t)
            idx = np.clip(idx, 0, len(pitch_f0) - 1)
            vmask = pitch_f0[idx] > 0
        if vmask.sum() < 3:
            vmask = (P.sum(0) > np.percentile(P.sum(0), 50))  # fallback: energetic frames

        # spectral moments (per frame over freq distribution, weighted by magnitude)
        eps = 1e-12
        Sn = S / (S.sum(0, keepdims=True) + eps)
        cent = (freqs[:, None] * Sn).sum(0)
        var = ((freqs[:, None] - cent[None, :]) ** 2 * Sn).sum(0)
        sd = np.sqrt(var) + eps
        skew = (((freqs[:, None] - cent[None, :]) ** 3) * Sn).sum(0) / (sd ** 3)
        kurt = (((freqs[:, None] - cent[None, :]) ** 4) * Sn).sum(0) / (sd ** 4)
        Pn = P / (P.sum(0, keepdims=True) + eps)
        ent = -(Pn * np.log(Pn + eps)).sum(0) / math.log(P.shape[0])  # normalized [0,1]
        vm = vmask if vmask.sum() >= 3 else np.ones(S.shape[1], bool)
        out["spectral_skewness"] = float(np.nanmean(skew[vm]))
        out["spectral_kurtosis"] = float(np.nanmean(kurt[vm]))
        out["spectral_entropy"]  = float(np.nanmean(ent[vm]))
        # rolloff (0.95)
        roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
        out["spectral_rolloff"] = float(np.nanmean(roll[vm]))
        # flux (frame-to-frame Euclidean of L2-normalized magnitude); dynamic -> mean
        Sl2 = S / (np.linalg.norm(S, axis=0, keepdims=True) + eps)
        flux = np.sqrt(((np.diff(Sl2, axis=1)) ** 2).sum(0))
        out["spectral_flux"] = float(np.nanmean(flux))
        # band energies (sum power over voiced frames)
        def bandE(lo, hi):
            b = (freqs >= lo) & (freqs < hi)
            return float(P[b][:, vm].sum())
        e_50_1k = bandE(50, 1000); e_1k_5k = bandE(1000, 5000)
        e_lo1k = bandE(0, 1000);   e_hi3k = bandE(3000, sr / 2)
        if e_50_1k > 0 and e_1k_5k > 0:
            out["alpha_ratio"] = float(10 * np.log10(e_1k_5k / e_50_1k))   # 1-5k vs 50-1000 (paper)
        if e_lo1k > 0 and e_hi3k > 0:
            out["LHR"] = float(10 * np.log10(e_lo1k / e_hi3k))             # <1k vs >3k (paper)
        # SPI: low (70-1600) vs high (1600-4500) harmonic energy, dB
        e_spi_lo = bandE(70, 1600); e_spi_hi = bandE(1600, 4500)
        if e_spi_lo > 0 and e_spi_hi > 0:
            out["SPI"] = float(10 * np.log10(e_spi_lo / e_spi_hi))
    except Exception:
        P = None; freqs = None; vmask = None

    # ---- RMS ----
    try:
        rms = librosa.feature.rms(y=x, frame_length=win, hop_length=hop)[0]
        out["RMS"] = float(np.sqrt(np.mean(x ** 2)))
    except Exception:
        pass

    # ---- AMD: depth of slow (<20 Hz) amplitude modulation ----
    try:
        fr = 0.005  # 5 ms env hop -> 200 Hz env rate
        eh = int(fr * sr)
        env = librosa.feature.rms(y=x, frame_length=2 * eh, hop_length=eh)[0]
        env_sr = sr / eh
        # active region only
        thr = np.percentile(env, 40)
        act = env[env > thr]
        if act.size > 10:
            b, a = ss.butter(2, min(20.0, env_sr / 2 - 1) / (env_sr / 2), btype="low")
            env_lp = ss.filtfilt(b, a, env)
            actlp = env_lp[env > thr]
            mu = np.mean(actlp)
            if mu > 0:
                out["AMD"] = float(np.std(actlp) / mu)  # coeff of variation of slow envelope
    except Exception:
        pass

    # ---- CPP / dCPP (per voiced frame cepstral peak prominence) ----
    try:
        if P is not None and vmask is not None and vmask.sum() >= 3:
            logP = 10 * np.log10(P + 1e-12)            # dB log-power, shape [F, T]
            ceps = np.fft.irfft(logP, axis=0)           # real cepstrum, quefrency along axis0
            nq = ceps.shape[0]
            quef = np.arange(nq) / sr
            qlo, qhi = 1.0 / f0max, 1.0 / f0min
            band = (quef >= qlo) & (quef <= qhi)
            regband = (quef >= 0.001) & (quef <= quef[nq // 2].max() if False else quef <= (nq // 2) / sr)
            cpps = []
            qb = quef[band]
            for ti in np.where(vmask)[0]:
                c = ceps[:, ti]
                # regression baseline over quefrency 1ms..nyquist-quef
                rb = (quef >= 0.001)
                A = np.vstack([quef[rb], np.ones(rb.sum())]).T
                coef, *_ = np.linalg.lstsq(A, c[rb], rcond=None)
                cb = c[band]
                pk = np.argmax(cb)
                base = coef[0] * qb[pk] + coef[1]
                cpps.append(cb[pk] - base)
            cpps = np.array(cpps)
            if cpps.size:
                out["CPP"] = float(np.nanmean(cpps))
                if cpps.size > 1:
                    out["dCPP"] = float(np.nanmean(np.abs(np.diff(cpps))))  # dynamic -> mean |Δ|
    except Exception:
        pass

    # ---- IHI: mean fractional deviation of partials from k*F0 ----
    try:
        if P is not None and freqs is not None and vmask is not None and f0v.size:
            devs = []
            Pm = P
            for ti in np.where(vmask)[0]:
                # frame F0 via nearest pitch
                tt = stft_t[ti]
                pj = np.clip(np.searchsorted(pitch_t, tt), 0, len(pitch_f0) - 1)
                f0i = pitch_f0[pj]
                if f0i <= 0: continue
                spec = Pm[:, ti]
                K = int(min(20, (sr / 2) / f0i))
                for k in range(1, K + 1):
                    fk = k * f0i
                    w = (freqs >= fk - 0.5 * f0i) & (freqs <= fk + 0.5 * f0i)
                    if w.sum() < 2: continue
                    fpk = freqs[w][np.argmax(spec[w])]
                    devs.append(abs(fpk - fk) / f0i)
            if devs:
                out["IHI"] = float(np.mean(devs))
    except Exception:
        pass

    # ---- SHR: subharmonic-to-harmonic amplitude ratio (approx, Sun-style) ----
    try:
        if P is not None and freqs is not None and vmask is not None and f0v.size:
            ratios = []
            amp = np.sqrt(P)
            for ti in np.where(vmask)[0]:
                tt = stft_t[ti]
                pj = np.clip(np.searchsorted(pitch_t, tt), 0, len(pitch_f0) - 1)
                f0i = pitch_f0[pj]
                if f0i <= 0: continue
                spec = amp[:, ti]
                def amp_at(fq):
                    j = int(round(fq / (sr / 2) * (len(freqs) - 1)))
                    if 0 <= j < len(freqs): return spec[j]
                    return 0.0
                Kmax = int(min(15, (sr / 2) / f0i))
                H = sum(amp_at(k * f0i) for k in range(1, Kmax + 1))
                SH = sum(amp_at((k - 0.5) * f0i) for k in range(1, Kmax + 1))
                if H > 0:
                    ratios.append(SH / H)
            if ratios:
                out["SHR"] = float(np.mean(ratios))
    except Exception:
        pass

    # ---- GNE: glottal-to-noise excitation (Michaelis, approx) ----
    try:
        g10 = librosa.resample(x, orig_sr=sr, target_sr=10000)
        a = librosa.lpc(g10 * np.hanning(len(g10)), order=13)
        res = ss.lfilter(a, [1.0], g10)
        centers = np.arange(1000, 4501, 500)
        bw = 1000.0; fsr = 10000.0
        envs = {}
        for fc in centers:
            lo = max(1.0, fc - bw / 2) / (fsr / 2); hi = min(fsr / 2 - 1, fc + bw / 2) / (fsr / 2)
            if hi <= lo: continue
            bb, aa = ss.butter(4, [lo, hi], btype="band")
            band = ss.filtfilt(bb, aa, res)
            envs[fc] = np.abs(ss.hilbert(band))
        best = 0.0
        cs = sorted(envs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if abs(cs[i] - cs[j]) < bw / 2:  # only sufficiently separated bands
                    continue
                e1 = envs[cs[i]] - np.mean(envs[cs[i]]); e2 = envs[cs[j]] - np.mean(envs[cs[j]])
                d = np.sqrt(np.sum(e1 ** 2) * np.sum(e2 ** 2))
                if d > 0:
                    cc = np.max(ss.correlate(e1, e2, mode="full", method="fft")) / d
                    best = max(best, cc)
        if best > 0:
            out["GNE"] = float(best)
    except Exception:
        pass

    # ---- Glottal-flow features via IAIF + GCIs: NAQ, CQ, GCT, MFDR, SQ ----
    try:
        if pp is not None and f0v.size:
            # GCIs
            npt = call(pp, "Get number of points")
            gci = np.array([call(pp, "Get time from index", i + 1) for i in range(int(npt))])
            gci = gci[np.isfinite(gci)]
            if gci.size >= 6:
                xv = x - np.mean(x)
                w = np.hanning(len(xv))
                a_g1 = librosa.lpc(xv * w, order=1)
                x1 = ss.lfilter(a_g1, [1.0], xv)
                a_vt1 = librosa.lpc(x1 * w, order=18)
                gest = ss.lfilter(a_vt1, [1.0], xv)
                g1 = ss.lfilter([1.0], [1.0, -0.99], gest)
                a_g2 = librosa.lpc(g1 * w, order=4)
                x2 = ss.lfilter(a_g2, [1.0], xv)
                a_vt2 = librosa.lpc(x2 * w, order=18)
                gd = ss.lfilter(a_vt2, [1.0], xv)          # flow derivative
                g = ss.lfilter([1.0], [1.0, -0.99], gd)    # flow
                naq, cq, sq, mfdr, gct = [], [], [], [], []
                for i in range(len(gci) - 1):
                    s0 = int(round(gci[i] * sr)); s1 = int(round(gci[i + 1] * sr))
                    T0 = gci[i + 1] - gci[i]
                    if T0 < 1.0 / f0max or T0 > 1.0 / f0min: continue
                    if s1 - s0 < 8 or s1 > len(g): continue
                    fl = g[s0:s1].copy()
                    fl = fl - np.min(fl)
                    fac = np.max(fl)
                    if fac <= 0: continue
                    fln = fl / fac                          # amplitude-normalized flow (scale-invariant)
                    dvn = np.diff(fln) * sr                 # time-derivative (1/s)
                    dmin = np.min(dvn)
                    if dmin >= 0: continue
                    mfdr_c = -dmin                          # max flow declination rate (1/s)
                    naq_c = 1.0 / (mfdr_c * T0)             # = f_ac/(d_peak*T0), f_ac=1
                    open_mask = fln > 0.2
                    cq_c = 1.0 - open_mask.mean()           # closed quotient
                    pk = np.argmax(fln)
                    oi = np.where(open_mask)[0]
                    if oi.size >= 3:
                        op = max(1, pk - oi[0]); cl = max(1, oi[-1] - pk)
                        sq_c = op / cl
                    else:
                        sq_c = np.nan
                    # physiological QC bounds
                    if 0.04 <= naq_c <= 0.6 and 0.1 <= cq_c <= 0.9:
                        naq.append(naq_c); cq.append(cq_c); gct.append(cq_c * T0)
                        mfdr.append(mfdr_c)
                        if np.isfinite(sq_c) and 0.2 <= sq_c <= 6: sq.append(sq_c)
                if len(naq) >= 5:
                    out["NAQ"] = float(np.median(naq))
                    out["CQ"]  = float(np.median(cq))
                    out["GCT"] = float(np.median(gct))
                    out["MFDR"] = float(np.median(mfdr))
                    if len(sq) >= 5:
                        out["SQ"] = float(np.median(sq))
    except Exception:
        pass

    # ---- phone-alignment features: VOT, speech_rate, BGD, SSPF ----
    phn = read_phn(os.path.splitext(path)[0] + ".PHN")
    if phn:
        try:
            total = phn[-1][1] - phn[0][0]
            # speaking region = remove leading/trailing + internal silences for rate/bgd
            # speech_rate: nuclei per second over non-silence duration
            speech_samp = sum((e - s) for s, e, lab in phn if lab not in SILENCE)
            n_nuclei = sum(1 for s, e, lab in phn if lab in NUCLEI)
            if speech_samp > 0:
                out["speech_rate"] = float(n_nuclei / (speech_samp / sr))
            # BGD: mean duration (s) of contiguous non-silence runs (breath groups)
            runs = []; cur = 0
            for s, e, lab in phn:
                if lab in SILENCE:
                    if cur > 0: runs.append(cur); cur = 0
                else:
                    cur += (e - s)
            if cur > 0: runs.append(cur)
            if runs:
                out["BGD"] = float(np.mean(runs) / sr)
            # VOT: mean duration (ms) of voiceless stop release segments p/t/k
            vots = [(e - s) / sr * 1000.0 for s, e, lab in phn if lab in VLESS_STOP_REL]
            if vots:
                out["VOT"] = float(np.mean(vots))
            # SSPF: spectral peak freq (Hz) of /s,sh,z,zh/ frication
            peaks = []
            for s, e, lab in phn:
                if lab in SIBILANTS and (e - s) > 320:
                    seg = x[s:e]
                    if seg.size < 256: continue
                    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                    fq = np.fft.rfftfreq(len(seg), 1.0 / sr)
                    hi = fq >= 1000
                    if hi.sum() > 0:
                        peaks.append(fq[hi][np.argmax(sp[hi])])
            if peaks:
                out["SSPF"] = float(np.mean(peaks))
        except Exception:
            pass

    return out


def list_wavs(root=TIMIT_ROOT):
    files = []
    for sub in ("TRAIN", "TEST"):
        files += glob.glob(os.path.join(root, sub, "DR*", "*", "*.WAV"))
    return sorted(files)


def main():
    import pandas as pd
    from multiprocessing import Pool
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    files = list_wavs()
    if not files:
        raise SystemExit(
            f"[extract] ABORT: no .WAV files found under VU_TIMIT_ROOT={TIMIT_ROOT!r}.\n"
            f"  Point VU_TIMIT_ROOT at the directory that DIRECTLY contains TRAIN/ and TEST/\n"
            f"  (e.g. .../timit/TIMIT), and make sure the TIMIT audio is actually unpacked\n"
            f"  there (if you only have timit_LDC93S1.tgz, extract it first:\n"
            f"    tar -xzf timit_LDC93S1.tgz   ->  creates timit/TIMIT/{{TRAIN,TEST}}/...).")
    print(f"[extract] {len(files)} utterances; seed={SEED}", flush=True)

    rows = []
    nproc = 2
    done = 0
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(extract_one, files, chunksize=8):
            rows.append(r); done += 1
            if done % 250 == 0:
                print(f"[extract] {done}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    print(f"[extract] done: {len(df)} rows", flush=True)

    # wide -> save raw wide for reuse
    df.to_parquet(os.path.join(out_dir, "_features_wide.parquet"), index=False)

    # long format: speaker_id, sex, utt_id, feature, value
    meta_cols = ["_speaker_id", "_sex", "_utt_id"]
    long = df.melt(id_vars=meta_cols, value_vars=FEATURE_ORDER,
                   var_name="feature", value_name="value")
    long = long.rename(columns={"_speaker_id": "speaker_id", "_sex": "sex", "_utt_id": "utt_id"})
    long.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)

    # coverage.csv
    n = len(df)
    cov_rows = []
    for f in FEATURE_ORDER:
        nz = int(df[f].notna().sum())
        frac = nz / n if n else 0.0
        status = "MEASURED" if frac > 0 else "NOT MEASURED"
        note = NOT_MEASURED_BY_DESIGN.get(f, "")
        cov_rows.append({"feature": f, "category": FEATURE_CATEGORY[f],
                         "n_ok": nz, "n_total": n, "coverage": round(frac, 4),
                         "status": status, "note": note})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(out_dir, "coverage.csv"), index=False)
    print(cov.to_string(index=False), flush=True)
    print(f"[extract] MEASURED features: {(cov.coverage>0).sum()}/{len(cov)}", flush=True)


if __name__ == "__main__":
    main()
