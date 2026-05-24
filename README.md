# OpenHand

Point a webcam at your hand, get text back. OpenHand reads American Sign Language fingerspelling and transcribes it in the browser. MediaPipe extracts landmarks; ONNX models classify them via WebAssembly. No backend, no API keys.

Two recognition paths:

1. **Per-letter detection.** Hold up an A-Z handshape (plus `del` and `space`); the live display shows the predicted letter and confidence. Backed by a small MLP (~62K params) trained on a multi-signer ASL alphabet dataset.
2. **Record-to-phrase transcription.** Press Record, fingerspell a word or phrase, press Stop. The full landmark buffer goes through a Squeezeformer + CTC model trained on the [Kaggle ASL Fingerspelling competition data](https://www.kaggle.com/competitions/asl-fingerspelling/) and the decoded text drops into the output bar.

The Learn-the-letters view shows a reference image per A-Z (plus `del` / `space`) and grades your handshape against the target letter live. J and Z are motion letters; the Learn view scores them against the CTC decode instead of the static MLP.

Train the models in the sibling repo [openhand-model](https://www.github.com/catherinepereira/openhand-model) and copy the outputs into `public/models/`. See "Setting up the models" below.

# Running

```
npm install
npm run dev
```

## Known limitations

- J and Z need motion to disambiguate from I and D. The static-frame MLP gets them wrong some of the time; the CTC model handles them fine because it sees a window of frames.
- The MLP is rotation-sensitive (it sees raw camera-frame landmarks), so signs whose handshape depends on orientation (P, G, H) may need exaggerated angles to register.
- MediaPipe's handedness label is camera-POV, not user-POV. A right-handed signer with a non-mirrored feed has their right hand labeled "Left". The classifier picks "Right" (camera-POV) as the dominant hand by convention.
- Only one hand feeds the alphabet classifier. `numHands` is 2 in MediaPipe but only the right hand reaches the MLP.
- `del` and `space` are static-frame classes only. `del` is really a swipe gesture in ASL; the MLP scores the peak pose of the swipe, which is less reliable than holding a static letter.
- WebAssembly ONNX is ~2-3x slower than native; the CTC beam search runs in ~20-50ms in the browser (vs ~5ms native). Per-letter inference stays well under 5ms.

## License

MIT
