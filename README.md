# OpenHand

<img width="600" src="https://i.ibb.co/Cpp2CvdS/Screenshot-2026-06-08-134011.png" alt="OpenHand" />

Point a webcam at your hand and get text back. OpenHand reads American Sign Language fingerspelling and transcribes it in the browser. MediaPipe extracts landmarks and ONNX models classify them.

Two recognition paths:

1. **Per-letter detection.** Hold up an A-Z handshape (plus `del` and `space`) and the live display will show the predicted letter and confidence. Individual letters are classified by a small MLP (~62K params) trained on a multi-signer ASL alphabet dataset.
2. **Record-to-phrase transcription.** Press Record, fingerspell a word or phrase, press Stop. The full landmark buffer goes through a Squeezeformer + CTC model trained on the [Kaggle ASL Fingerspelling competition data](https://www.kaggle.com/competitions/asl-fingerspelling/) and the decoded text is added to the output bar.

The Learn-the-letters view shows a reference image per A-Z (plus `del` / `space`) and grades your handshape against the target letter live. J and Z are motion letters so the Learn view scores them against the CTC decode instead of the static MLP.

Train the models in the sibling repo [openhand-model](https://www.github.com/catherinepereira/openhand-model) and copy the outputs into `public/models/` to use.

# Running

```bash
npm install
npm run dev
```

## Known limitations

- J and Z need motion to disambiguate from I and D. The static-frame MLP gets them wrong some of the time while the CTC model handles them fine because it sees a window of frames.
- The MLP is rotation-sensitive (it sees raw camera-frame landmarks), so signs whose handshape depends on orientation (P, G, H) may need exaggerated angles to register.
- Only one hand feeds the alphabet classifier. `numHands` is 2 in MediaPipe but only the right hand reaches the MLP (for now).
- `del` and `space` are static-frame classes only. `del` is really a swipe gesture in ASL; the MLP scores the peak pose of the swipe, which is less reliable than holding a static letter.

## License

MIT. See [LICENSE](LICENSE).
