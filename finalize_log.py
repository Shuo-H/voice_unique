import time, importlib
mods = ['sphfile','parselmouth','librosa','sklearn','numpy','scipy','pandas','pyarrow']
with open("run.log","a") as fh:
    fh.write("--- RUN_END ---\n")
    fh.write("RUN_END_UTC: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    for m in mods:
        try:
            v = getattr(importlib.import_module(m), "__version__", "?")
        except Exception as e:
            v = "ERR:" + str(e)[:40]
        fh.write(f"lib {m} {v}\n")
    fh.write("extract_wallclock_sec=4949.0 (6300 utts, 16 procs)\n")
    fh.write("stages_done: extract, analyze(1-5,7), classify(6), report\n")
print("log finalized")
