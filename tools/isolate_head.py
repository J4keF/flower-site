"""Keep only the flower head in a sequence of transparent frames.

Leaves, sepals and stems are removed (green hues, then anything thinner than
the head via a morphological opening), the largest remaining blob is kept,
and edges are restored from the original alpha.

Each frame is then shifted so the head's lowest point sits on the bottom
edge (it rests on the page and grows upward) with its centre held steady,
and all frames are written at one shared size.

usage: isolate_head.py <in_dir> <out_dir> [radius_frac=0.025] [bottom|center]
       isolate_head.py --realign <in_dir> <out_dir> <bottom|center>   (frames already masked)
"""
import os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r

def head_mask(rgba, radius):
    a = rgba[..., 3].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) / 255
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # green dominates both other channels by a clear margin
    green = (g > r * 1.08) & (g > b * 1.02) & (sat > 0.15)

    solid = (a > 0.35) & ~green

    # find the head at half resolution (the big opening is the slow part)
    half = Image.fromarray(solid.astype(np.uint8) * 255).resize(
        (solid.shape[1] // 2, solid.shape[0] // 2), Image.BILINEAR)
    opened = ndimage.binary_opening(np.asarray(half) > 127, structure=disk(max(1, radius // 2)))
    labels, n = ndimage.label(opened)
    if n == 0:
        return np.zeros_like(a)
    sizes = ndimage.sum(opened, labels, range(1, n + 1))
    core = ndimage.binary_fill_holes(labels == (np.argmax(sizes) + 1))
    core = np.asarray(Image.fromarray(core.astype(np.uint8) * 255).resize(
        (solid.shape[1], solid.shape[0]), Image.BILINEAR)) > 127

    # restore petal edges near the core, but drop wisps a few px wide
    near = solid & ndimage.binary_dilation(core, structure=disk(radius + 2))
    keep = ndimage.binary_opening(near | core, structure=disk(2)) | core
    labels, _ = ndimage.label(keep)
    keep = np.isin(labels, np.unique(labels[core]))
    keep = ndimage.binary_dilation(keep, structure=disk(2))  # room for anti-aliased rims
    soft = ndimage.gaussian_filter(keep.astype(np.float32), 1.0)
    # leaf showing through anti-aliased rims: clear any clearly green pixel
    rim_green = (g > r * 1.05) & (g > b * 1.05) & (sat > 0.12)
    return a * soft * ~rim_green

def align(frames, anchor="bottom", pad=4):
    """Re-place frames on one shared canvas, held steady by a smoothed track.

    anchor="bottom": each head rests on the bottom edge and grows upward.
    anchor="center": each head's centre sits at the canvas centre and the
    bloom opens outward from it.
    """
    boxes = []
    for f in frames:
        ys, xs = np.nonzero(f[..., 3] > 20)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    boxes = np.array(boxes, np.float32)
    cx = ndimage.uniform_filter1d((boxes[:, 0] + boxes[:, 2]) / 2, 9, mode="nearest")
    cy = ndimage.uniform_filter1d((boxes[:, 1] + boxes[:, 3]) / 2, 9, mode="nearest")
    W = int(np.ceil(2 * (np.max(np.maximum(cx - boxes[:, 0], boxes[:, 2] - cx)) + pad)))
    if anchor == "center":
        H = int(np.ceil(2 * (np.max(np.maximum(cy - boxes[:, 1], boxes[:, 3] - cy)) + pad)))
    else:
        H = int(np.ceil(np.max(boxes[:, 3] - boxes[:, 1]) + pad))
    out = []
    for f, box, x, y in zip(frames, boxes, cx, cy):
        top = H / 2 - y if anchor == "center" else H - box[3]
        canvas = Image.new("RGBA", (W, H))
        canvas.paste(Image.fromarray(f), (int(round(W / 2 - x)), int(round(top))))
        out.append(canvas)
    return out

if __name__ == "__main__":
    args = sys.argv[1:]
    realign = args[0] == "--realign"
    if realign:
        args = args[1:]
    src, out = args[0], args[1]
    frac = 0.025 if realign or len(args) < 3 else float(args[2])
    anchor = args[2] if realign else (args[3] if len(args) > 3 else "bottom")
    os.makedirs(out, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.endswith((".webp", ".png")))
    frames = []
    for f in names:
        rgba = np.asarray(Image.open(os.path.join(src, f)).convert("RGBA")).copy()
        if not realign:
            radius = max(2, round(rgba.shape[1] * frac))
            rgba[..., 3] = (head_mask(rgba, radius) * 255).astype(np.uint8)
        frames.append(rgba)
    for f, im in zip(names, align(frames, anchor)):
        im.save(os.path.join(out, os.path.splitext(f)[0] + ".webp"), quality=86, method=6, exact=True)
    print(f"{len(names)} frames, {im.width}x{im.height}")
