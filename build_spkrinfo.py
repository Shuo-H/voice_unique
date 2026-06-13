"""Parse TIMIT DOC/SPKRINFO.TXT -> results/speaker_meta.csv.
Provides sex, height (cm), age (yr) as shared-parent / body-size proxies
for the parent-residual effective-dimensionality analysis. READ-ONLY corpus."""
import os, csv, re, datetime

CORPUS = r"C:\Users\shuoo\Desktop\voice_unique\data\TIMIT"
SRC = os.path.join(CORPUS, "DOC", "SPKRINFO.TXT")


def parse_height(s):
    m = re.match(r"(\d+)'(\d+)\"?", s)
    if not m:
        return None
    feet, inch = int(m.group(1)), int(m.group(2))
    return round((feet * 12 + inch) * 2.54, 1)  # cm


def parse_date(s):
    # MM/DD/YY
    try:
        mo, da, yr = s.split("/")
        yr = int(yr)
        yr += 1900 if yr > 25 else 2000  # birthdates 1930s-60s; recdates 1986
        return datetime.date(yr, int(mo), int(da))
    except Exception:
        return None


rows = []
with open(SRC) as f:
    for line in f:
        if line.startswith(";") or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        sid, sex, dr, use, recdate, birth, ht = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        speaker = sex + sid  # dir-name convention M/F + ID
        ht_cm = parse_height(ht)
        rd, bd = parse_date(recdate), parse_date(birth)
        age = None
        if rd and bd:
            age = round((rd - bd).days / 365.25, 1)
        rows.append({"speaker": speaker, "sex": sex, "dr": "DR" + dr,
                     "height_cm": ht_cm if ht_cm else "",
                     "age_yr": age if age else ""})

os.makedirs("results", exist_ok=True)
out = "results/speaker_meta.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["speaker", "sex", "dr", "height_cm", "age_yr"])
    w.writeheader()
    w.writerows(rows)

nh = sum(1 for r in rows if r["height_cm"] != "")
na = sum(1 for r in rows if r["age_yr"] != "")
print(f"speakers={len(rows)} with_height={nh} with_age={na} -> {out}")
