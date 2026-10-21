# jakefogel.com — bloom landing page

A single static page: an off-white background with a flower in the center that
blooms as a time-lapse on load, slows, and settles into a still. Each refresh
picks a random flower (`?flower=rose` pins one).

## Layout

- `index.html` — the whole page (canvas player, easing, flower list)
- `flowers/<id>/000–063.webp` — transparent frames, bud first, full bloom last
- `tools/` — the pipeline that turns black-background time-lapses into frames
- `dahlia.webp`, `dahlia.png` — the earlier still dahlia (not used by the page)

## Adding a flower

1. Extract frames: `ffmpeg -i clip.mp4 -an work/frames/%04d.png`
2. Build: `python3 tools/build_flower.py work/frames flowers/<id> x0,y0,x1,y1 64 <size> [l,t,r,b fade px]`
   (needs `pip install "rembg[cpu]" scipy pillow numpy onnxruntime`; BiRefNet is
   ~1 GB on first run and peaks around 8 GB RAM)
3. Add `{ id, name, frames: 64, size }` to `FLOWERS` in `index.html`
4. Check it: `npx serve .` then `node tools/verify.mjs http://localhost:3000/ <id> shot`

## Footage credits

- Crocus — "Purple flower blooming slowly", Mixkit (Mixkit free license)
- Rose — "Time-lapse of opening red rose isolated on black background" by Alex Tr,
  [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Time-lapse_of_opening_red_rose_isolated_on_black_background.Blooming_red_roses_flower_buds.webm) (credited on the page)
