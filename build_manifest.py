"""Build the TIMIT utterance manifest. Corpus is READ-ONLY."""
import os, glob, csv, sys

CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"
OUT = os.path.join("results", "manifest.csv")

rows = []
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
            sex = spk[0]  # M or F
            for wav in sorted(glob.glob(os.path.join(spkp, "*.WAV"))):
                utt = os.path.splitext(os.path.basename(wav))[0]
                phn = os.path.splitext(wav)[0] + ".PHN"
                rows.append({
                    "split": split,
                    "dialect": dr,
                    "speaker": spk,
                    "sex": sex,
                    "utt": utt,
                    "utt_id": f"{spk}_{utt}",
                    "wav": wav,
                    "phn": phn if os.path.isfile(phn) else "",
                })

os.makedirs("results", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

speakers = sorted(set(r["speaker"] for r in rows))
males = sorted(set(r["speaker"] for r in rows if r["sex"] == "M"))
females = sorted(set(r["speaker"] for r in rows if r["sex"] == "F"))
missing_phn = sum(1 for r in rows if not r["phn"])
print(f"utterances={len(rows)}")
print(f"speakers={len(speakers)} (M={len(males)} F={len(females)})")
print(f"missing_PHN={missing_phn}")
print(f"wrote {OUT}")
