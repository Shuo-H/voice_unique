"""Build a SMALL balanced subset manifest + full speaker_meta for the smoke test.
Writes results/manifest.csv and results/speaker_meta.csv RELATIVE TO CWD
(run this with cwd = the smoke working dir). Corpus is READ-ONLY.

Subset = first N male + first N female speakers (sorted by speaker id), all 10
utterances each. Override N with env var SMOKE_N_PER_SEX (default 8)."""
import os, glob, csv, sys, re, datetime

CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"
N_PER_SEX = int(os.environ.get("SMOKE_N_PER_SEX", "8"))

# ---- walk corpus, collect every utterance row (same schema as build_manifest) ----
allrows = []
for split in ("TRAIN", "TEST"):
    base = os.path.join(CORPUS, split)
    if not os.path.isdir(base):
        print(f"FATAL: missing split dir {base}")
        sys.exit(2)
    for dr in sorted(os.listdir(base)):
        drp = os.path.join(base, dr)
        if not (os.path.isdir(drp) and dr.startswith("DR")):
            continue
        for spk in sorted(os.listdir(drp)):
            spkp = os.path.join(drp, spk)
            if not os.path.isdir(spkp):
                continue
            sex = spk[0]
            for wav in sorted(glob.glob(os.path.join(spkp, "*.WAV"))):
                utt = os.path.splitext(os.path.basename(wav))[0]
                phn = os.path.splitext(wav)[0] + ".PHN"
                allrows.append({"split": split, "dialect": dr, "speaker": spk,
                                "sex": sex, "utt": utt, "utt_id": f"{spk}_{utt}",
                                "wav": wav, "phn": phn if os.path.isfile(phn) else ""})

males = sorted({r["speaker"] for r in allrows if r["sex"] == "M"})[:N_PER_SEX]
females = sorted({r["speaker"] for r in allrows if r["sex"] == "F"})[:N_PER_SEX]
chosen = set(males) | set(females)
rows = [r for r in allrows if r["speaker"] in chosen]

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"SUBSET speakers={len(chosen)} (M={len(males)} F={len(females)}) "
      f"utts={len(rows)} missing_PHN={sum(1 for r in rows if not r['phn'])}")

# ---- full speaker_meta (same logic as build_spkrinfo); extra rows are harmless ----
def parse_height(s):
    m = re.match(r"(\d+)'(\d+)\"?", s)
    if not m:
        return None
    return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)

def parse_date(s):
    try:
        mo, da, yr = s.split("/"); yr = int(yr)
        yr += 1900 if yr > 25 else 2000
        return datetime.date(yr, int(mo), int(da))
    except Exception:
        return None

meta = []
with open(os.path.join(CORPUS, "DOC", "SPKRINFO.TXT")) as f:
    for line in f:
        if line.startswith(";") or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        sid, sex, dr, use, recdate, birth, ht = p[:7]
        ht_cm = parse_height(ht); rd, bd = parse_date(recdate), parse_date(birth)
        age = round((rd - bd).days / 365.25, 1) if (rd and bd) else None
        meta.append({"speaker": sex + sid, "sex": sex, "dr": "DR" + dr,
                     "height_cm": ht_cm if ht_cm else "",
                     "age_yr": age if age else ""})
with open(os.path.join("results", "speaker_meta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["speaker", "sex", "dr", "height_cm", "age_yr"])
    w.writeheader(); w.writerows(meta)
print(f"speaker_meta rows={len(meta)} -> results/speaker_meta.csv")
