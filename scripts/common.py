"""Shared helpers: corpus root, the 40 canonical feature names, speaker metadata parsing, utterance enumeration."""
import os, re, datetime

CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"
SEED = 1234

# THE 40 CANONICAL FEATURES (VTLE removed). Order is fixed and authoritative.
FEATURES_GLOTTAL = ["F0", "jitter", "shimmer", "GCT", "CQ", "MFDR", "SQ", "NAQ", "SHR", "IHI", "VFP", "semitone_SD_F0"]
FEATURES_VT      = ["F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5", "Nasality"]
FEATURES_SPEC    = ["spectral_skewness", "spectral_kurtosis", "spectral_entropy", "spectral_rolloff",
                    "spectral_flux", "alpha_ratio", "LHR", "SPI", "GNE", "SSPF"]
FEATURES_ARTIC   = ["CPP", "dCPP", "RMS", "AMD", "speech_rate", "VOT", "BGD"]
FEATURES_40 = FEATURES_GLOTTAL + FEATURES_VT + FEATURES_SPEC + FEATURES_ARTIC
assert len(FEATURES_40) == 40, len(FEATURES_40)

NONSPEECH_PHONES = {"h#", "pau", "epi", "1", "2"}  # silence / non-phonetic markers
STOP_CLOSURES = {"bcl", "dcl", "gcl", "kcl", "pcl", "tcl"}
STOP_RELEASES = {"b", "d", "g", "k", "p", "t"}
VOWELS = {"iy","ih","eh","ey","ae","aa","aw","ay","ah","ao","oy","ow","uh","uw","ux","er","ax","ix","axr","ax-h"}


def parse_spkrinfo(corpus=CORPUS):
    """Return dict: speaker_id(e.g. 'FCJF0') -> {sex, dr, use, height_cm, age_years}."""
    path = os.path.join(corpus, "DOC", "SPKRINFO.TXT")
    out = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            sid, sex, dr, use, recdate, birthdate, ht = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            full_id = sex + sid  # directory convention {SEX}{ID}
            height_cm = _parse_height(ht)
            age = _parse_age(recdate, birthdate)
            out[full_id] = {"sex": sex, "dr": dr, "use": use,
                            "height_cm": height_cm, "age_years": age}
    return out


def _parse_height(ht):
    # forms like 5'11" or 5'5"
    m = re.match(r"(\d+)'(\d+)\"?", ht)
    if m:
        feet, inch = int(m.group(1)), int(m.group(2))
        return round((feet * 12 + inch) * 2.54, 1)
    return None


def _parse_age(recdate, birthdate):
    try:
        def pd(s):
            mm, dd, yy = s.split("/")
            yy = int(yy)
            yy += 1900 if yy > 30 else 2000
            return datetime.date(yy, int(mm), int(dd))
        rec, bir = pd(recdate), pd(birthdate)
        return round((rec - bir).days / 365.25, 1)
    except Exception:
        return None


def enumerate_utterances(corpus=CORPUS, smoke_per_sex_per_dr=None):
    """Yield dicts {utt_id, wav, phn, split, dr, speaker, sex}.
    If smoke_per_sex_per_dr is set, take at most that many M and that many F speakers per DR from TRAIN only."""
    items = []
    splits = ["TRAIN", "TEST"]
    if smoke_per_sex_per_dr is not None:
        splits = ["TRAIN"]
    for split in splits:
        split_dir = os.path.join(corpus, split)
        for dr in sorted(os.listdir(split_dir)):
            dr_dir = os.path.join(split_dir, dr)
            if not os.path.isdir(dr_dir):
                continue
            spk_dirs = sorted(os.listdir(dr_dir))
            if smoke_per_sex_per_dr is not None:
                chosen = []
                for sx in ("M", "F"):
                    sel = [s for s in spk_dirs if s.startswith(sx)][:smoke_per_sex_per_dr]
                    chosen += sel
                spk_dirs = chosen
            for spk in spk_dirs:
                spk_dir = os.path.join(dr_dir, spk)
                if not os.path.isdir(spk_dir):
                    continue
                sex = spk[0]
                prefixes = sorted({os.path.splitext(f)[0] for f in os.listdir(spk_dir) if f.endswith(".WAV")})
                for pre in prefixes:
                    wav = os.path.join(spk_dir, pre + ".WAV")
                    phn = os.path.join(spk_dir, pre + ".PHN")
                    items.append({
                        "utt_id": f"{split}_{dr}_{spk}_{pre}",
                        "wav": wav, "phn": phn,
                        "split": split, "dr": dr, "speaker": spk, "sex": sex,
                    })
    return items
