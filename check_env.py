import sys, importlib, platform

print("python", sys.version.split()[0], platform.platform())
mods = ['sphfile', 'parselmouth', 'librosa', 'sklearn', 'numpy', 'scipy', 'pandas', 'pyarrow']
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', '?')
        print(f"OK {m} {v}")
    except Exception as e:
        print(f"MISSING {m}: {e}")
        missing.append(m)
if missing:
    print("RESULT: MISSING_PACKAGES " + ",".join(missing))
    sys.exit(2)
else:
    print("RESULT: ALL_PRESENT")
