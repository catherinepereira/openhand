# OpenHand

Point a webcam at your hand, get text back. OpenHand reads American Sign Language fingerspelling in the browser and converts it to text and, if you want, speech.

Two recognition paths run in parallel:

1. **Per-frame letter detection.** Hold up an A-Z handshape (plus `del` and `space`); the app accumulates letters into a sliding output bar with `del` deleting the last character and `space` adding a space. Backed by a small MLP (~62K params) trained on a multi-signer ASL alphabet dataset.
2. **Streaming phrase transcription.** Sign a full fingerspelled word or phrase; a rolling buffer of the last few seconds is decoded by a Squeezeformer + CTC model trained on the [Kaggle ASL Fingerspelling competition data](https://www.kaggle.com/competitions/asl-fingerspelling/). Renders below the letter display when the buffer has any hand frames.

The Learn-the-letters view shows a reference image per A-Z (plus `del` / `space`) and grades your handshape against the target letter live. J and Z are motion letters; the Learn view scores them against the CTC decode instead of the static MLP.

MediaPipe Hand + Pose + Face Landmarkers run in the browser to extract landmarks. The trained ONNX models run server-side via `onnxruntime`. No GPU in the runtime path. Alphabet inference is ~0.02 ms per frame on CPU.

Train the models in the sibling repo [openhand-model](https://www.github.com/catherinepereira/openhand-model) and copy the outputs into `backend/models/artifacts/`. See "Setting up the models" below.


## Running

```
cd frontend
npm run dev
```

## Known limitations

- J and Z need motion to disambiguate from I and D. The static-frame
  MLP gets them wrong some of the time; the CTC model handles them
  fine because it sees a window of frames.
- The MLP is rotation-sensitive (it sees raw camera-frame landmarks),
  so signs whose handshape depends on orientation (P, G, H) may need
  exaggerated angles to register.
- MediaPipe's handedness label is camera-POV, not user-POV. A
  right-handed signer with a non-mirrored feed has their right hand
  labeled "Left". The classifier picks "Right" (camera-POV) as the
  dominant hand by convention.
- Only one hand feeds the classifier. `numHands` is 2 in MediaPipe but
  only the right hand reaches the MLP.
- `del` and `space` are static-frame classes only. `del` is really a
  swipe gesture in ASL; the MLP scores the peak pose of the swipe,
  which is less reliable than holding a static letter.
