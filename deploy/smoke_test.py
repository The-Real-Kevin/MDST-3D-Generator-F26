"""
Run before every deploy, from inside the assembled Space folder:

    cd space && python ../deploy/smoke_test.py

Proves the Space will actually work -- the trained model loads and is NOT
stale, a sentence turns into a city, the files get written, and the page
builds -- so a broken commit fails here on GitHub instead of on the live site.
Cleans up after itself so nothing it makes gets uploaded.
"""
import os
import shutil
import sys
from pathlib import Path

os.environ["CITYGEN_HOSTED"] = "1"      # behave exactly as on the Space
sys.path.insert(0, os.getcwd())
sys.argv = ["app.py"]

import app  # noqa: E402

failures = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"\n       {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


for path in (app.CHECKPOINT / "model.pt", app.DATASET):
    check(f"{path.relative_to(app.ROOT)} is present", path.exists(),
          "commit it to GitHub (the model needs it, and the staleness check needs the CSV)")

app.MODEL.load()
check("the text model loads and is fresh", app.MODEL.error is None, app.MODEL.error or "")

smoke_dir = app.GALLERY / "_smoke_test"
try:
    if app.MODEL.error is None:
        params = app.MODEL.params("a dense city in the hills with three business hubs")
        kwargs = app.to_generator_kwargs(params)
        print("   predicted:", kwargs)
        paths = app.make_city(kwargs, 7, 64, "smoke test", prompt="smoke test", folder=smoke_dir)
        for ext in ("png", "stl", "glb"):
            size = Path(paths[ext]).stat().st_size
            check(f"city .{ext} written ({size:,} bytes)", size > 1000)
        check("the sidebar lists the new city", "smoke test" in app.gallery_html_for_folder(smoke_dir))
    app.build_ui()
    check("the page builds", True)
except Exception as e:  # report, then fail the deploy
    check("no crash", False, f"{type(e).__name__}: {e}")
finally:
    shutil.rmtree(smoke_dir, ignore_errors=True)       # only what this test made
    if app.GALLERY.exists() and not any(app.GALLERY.iterdir()):
        app.GALLERY.rmdir()

if failures:
    sys.exit(f"\n{len(failures)} check(s) failed -- not deploying.")
print("\nall good -- safe to deploy")
