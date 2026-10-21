"""Build a transparent frame sequence for one flower clip.

usage: build_flower.py <frames_dir> <out_dir> <x0,y0,x1,y1> <n_frames> [size] [fade l,t,r,b px] [radial r,band]

The crop may extend past the frame; the overhang is black (transparent). With
a px fade, fading is measured from the real frame edges inside the crop.
"""
import os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
from rembg import new_session, remove
from key import key_frame

src, out, crop, n = sys.argv[1], sys.argv[2], tuple(map(int, sys.argv[3].split(","))), int(sys.argv[4])
size = int(sys.argv[5]) if len(sys.argv) > 5 else None
fade = tuple(map(int, sys.argv[6].split(","))) if len(sys.argv) > 6 else 0.16
radial = tuple(map(float, sys.argv[7].split(","))) if len(sys.argv) > 7 else None
os.makedirs(out, exist_ok=True)
files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
picks = np.linspace(0, len(files) - 1, n).round().astype(int)
# the default CPU arena grows per inference until the process is OOM-killed
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False
session = new_session("birefnet-general", sess_opts=opts)

for k, idx in enumerate(picks):
    dst = os.path.join(out, f"{k:03d}.webp")
    if os.path.exists(dst):
        continue
    t = time.time()
    full = Image.open(os.path.join(src, files[idx])).convert("RGB")
    im = full.crop(crop)
    # where the real footage sits inside the (possibly overhanging) crop
    valid = (max(0, -crop[0]), max(0, -crop[1]),
             min(crop[2], full.width) - crop[0], min(crop[3], full.height) - crop[1])
    mask = remove(im, session=session, only_mask=True)
    frame = key_frame(im, mask, fade=fade, valid=valid, radial=radial)
    if size:
        frame = frame.resize((size, size), Image.LANCZOS)
    frame.save(dst, quality=86, method=6, exact=True)
    print(f"{k + 1}/{n} src={files[idx]} {time.time() - t:.1f}s", flush=True)
print("DONE", flush=True)
