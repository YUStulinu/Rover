"""Extract the ASL-format sequences from a downloaded EuRoC environment zip
(each holds, per sequence, a ROS bag and an inner zip in ASL format; only
the inner zips are extracted), then delete the outer zip."""
import os
import shutil
import sys
import zipfile

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "euroc")
outer = os.path.join(here, sys.argv[1])
with zipfile.ZipFile(outer) as z:
    for info in z.infolist():
        if not info.filename.endswith(".zip") or "/._" in info.filename:
            continue
        seq = os.path.basename(info.filename)[:-4]
        tmp = os.path.join(here, seq + ".inner.zip")
        with z.open(info) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        with zipfile.ZipFile(tmp) as inner:
            inner.extractall(os.path.join(here, seq))
        os.remove(tmp)
        print("extracted", seq, sorted(os.listdir(os.path.join(here, seq, "mav0"))), flush=True)
os.remove(outer)
