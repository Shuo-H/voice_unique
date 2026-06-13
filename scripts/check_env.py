"""Environment + version check. Writes to run.log. STOPs (exit 1) if a required package is missing."""
import sys, platform, time

REQUIRED = ["sphfile", "parselmouth", "librosa", "sklearn", "numpy", "scipy", "pandas", "pyarrow"]

def main():
    lines = []
    lines.append(f"=== check_env @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    lines.append(f"python = {sys.version.split()[0]} ({platform.platform()})")
    missing = []
    versions = {}
    for pkg in REQUIRED:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "?")
            versions[pkg] = v
            lines.append(f"  {pkg} = {v}")
        except Exception as e:
            missing.append(pkg)
            lines.append(f"  {pkg} = MISSING ({e})")
    out = "\n".join(lines) + "\n"
    with open("run.log", "a", encoding="utf-8") as f:
        f.write(out)
    print(out)
    if missing:
        print(f"STOP: missing required packages: {missing}", file=sys.stderr)
        sys.exit(1)
    print("ENV OK")

if __name__ == "__main__":
    main()
