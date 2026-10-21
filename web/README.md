# Browser demo: camera person tracking

`index.html` runs on a phone or desktop browser: camera frames go to
MediaPipe's EfficientDet-Lite0 person detector (WebGL GPU delegate, CPU
fallback), its boxes to this repository's 2D tracker compiled to WebAssembly
(`crates/factor-tracker-wasm`), and the confirmed tracks are drawn over the video
with their IDs. Nothing is uploaded; camera access needs HTTPS (or localhost).

Camera only for now: boxes and IDs in the image, no metric positions.

Run locally:

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p factor-tracker-wasm
cp target/wasm32-unknown-unknown/release/factor_tracker_wasm.wasm web/
python3 -m http.server -d web 8000   # http://localhost:8000
```

Published by `.github/workflows/pages.yml` on pushes to `main` once the
repository's Pages source is set to "GitHub Actions".

Third-party code and models loaded at run time: MediaPipe Tasks Vision
0.10.14 (Apache-2.0, from jsDelivr) and the EfficientDet-Lite0 COCO model
(Apache-2.0, from Google's MediaPipe model storage).

Next steps: run the detector in a Web Worker so detection and drawing
overlap, try a WebGPU detector (ONNX Runtime Web) for faster phones, and
estimate ground-plane positions from the phone's orientation sensors.
