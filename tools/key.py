"""Turn black-background frames into transparent RGBA frames.

Alpha is the brightness key (precise soft edges on pure black) combined with an
AI subject mask (keeps dark, shadowed parts of the flower opaque). Colours are
un-premultiplied against black so edges carry no dark fringe, the edge is
choked slightly to drop the codec's sharpening halo, and the crop border fades
out so leaves and stems don't end in a hard line.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def edge_fade(h, w, fade, valid=None):
    """fade: fraction of the crop, or (left, top, right, bottom) in px measured
    from the `valid` rect (the part of the crop that came from real footage)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if not isinstance(fade, tuple):
        d = np.minimum.reduce([xx / w, (w - 1 - xx) / w, yy / h, (h - 1 - yy) / h])
        return smoothstep(d / fade)
    x0, y0, x1, y1 = valid or (0, 0, w, h)
    out = np.ones((h, w), np.float32)
    for dist, px in ((xx - x0, fade[0]), (yy - y0, fade[1]), (x1 - 1 - xx, fade[2]), (y1 - 1 - yy, fade[3])):
        if px:
            out *= smoothstep(dist / px)
    return out

def key_frame(im, ai_mask=None, lo=14, hi=70, fade=0.16, choke=1, valid=None, radial=None):
    """im: cropped RGB frame; ai_mask: matching L-mode mask (or None)."""
    rgb = np.asarray(im).astype(np.float32)
    lum = np.asarray(im.filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32).max(2)
    a = smoothstep((lum - lo) / (hi - lo))

    if ai_mask is not None:
        m = np.asarray(ai_mask).astype(np.float32) / 255
        # trust the AI mask for the interior only; the luma key owns the edges
        core = ndimage.grey_erosion(m, size=(5, 5))
        a = np.maximum(a, core)

    # un-premultiply against black
    col = np.clip(rgb / np.maximum(a, 1e-3)[..., None], 0, 255)
    col = np.where(a[..., None] > 0.02, col, 0)

    if choke:
        a = ndimage.grey_erosion(a, size=(2 * choke + 1, 2 * choke + 1))
        a = ndimage.gaussian_filter(a, 0.6)

    a = a * edge_fade(*a.shape, fade, valid)
    if radial:  # (radius, band) as fractions of width: circular vignette
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(xx - w / 2, yy - h / 2) / w
        a = a * smoothstep((radial[0] - r) / radial[1])
    return Image.fromarray(np.dstack([col, a * 255]).astype(np.uint8), "RGBA")
