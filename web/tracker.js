// Loads crates/factor-tracker-wasm (built for wasm32-unknown-unknown) and wraps its
// scalar exports: ft_reset, ft_begin, ft_add, ft_end, ft_track.
const FIELDS = ["id", "cx", "cy", "w", "h", "state", "updated"];

export async function loadTracker(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
  const e = instance.exports;
  return {
    reset() {
      e.ft_reset();
    },
    // detections: [{x, y, w, h, score}] with (x, y) the top-left corner in
    // video pixels; returns [{id, cx, cy, w, h, state, updated}], state 0
    // tentative, 1 confirmed, 2 lost.
    step(timeMs, detections) {
      e.ft_begin(timeMs);
      for (const d of detections) e.ft_add(d.x + d.w / 2, d.y + d.h / 2, d.w, d.h, d.score);
      const n = e.ft_end();
      const tracks = [];
      for (let i = 0; i < n; i += 1) {
        const t = {};
        FIELDS.forEach((name, k) => { t[name] = e.ft_track(i, k); });
        t.updated = t.updated === 1;
        tracks.push(t);
      }
      return tracks;
    },
  };
}
