from glob import glob
import os

folders = glob(
    "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\mods/*"
)

for f in folders:
    if "isaac-env" not in f:
        open(os.path.join(f, "disable.it"), "w").close()
