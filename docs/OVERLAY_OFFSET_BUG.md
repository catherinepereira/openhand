# The skeleton overlay was offset, and the bug was in MediaPipe's input sampling

> A debugging post-mortem from porting OpenHand's MediaPipe pipeline
> from the Python backend to the browser. The fix is a one-liner. The
> hunt was three hours.

## What we saw

After moving MediaPipe Hand/Pose/Face detection from the Python backend
to the browser via `@mediapipe/tasks-vision`, the live hand-skeleton
overlay started rendering **slightly below and to the left** of the
user's actual hand. Same skeleton shape, same orientation, same scale —
just a constant pixel offset, with no obvious correlation to hand
position. The pre-port version, where the backend extracted landmarks
from base64 JPEGs, had no offset.

The hand classifier still worked (it doesn't depend on visual
alignment), but the overlay looked wrong. And the kind of wrong that
seems trivially fixable but never quite is.

## What we tried that didn't work

I burned a lot of debugging time on dead-end hypotheses:

- **Coordinate space flip wrong?** Drew the skeleton both with and
  without `x → 1 - x`. The flipped version was clearly closer to the
  hand. Not a flip problem.
- **`object-fit: cover` cropping?** Switched to `object-fit: fill`,
  switched between `transform: scaleX(-1)` on the canvas vs the video.
  No change.
- **Canvas backing buffer vs CSS display size?** Tried both — sizing
  the canvas to `getBoundingClientRect()` (the original working
  version) and sizing it to `videoWidth/videoHeight` (Medium articles
  recommend this). Same offset either way.
- **`numHands: 1` vs `numHands: 2`?** Same offset.
- **`runningMode: VIDEO` vs `IMAGE`?** Same offset.
- **Different MediaPipe initialisation flags?** Same offset.

Each test ate 5-10 minutes of "make the change, refresh, hold up your
hand, eyeball the skeleton." None of them mattered.

## The diagnostic that cracked it

The breakthrough came from a piece of code I almost didn't write: a
parallel comparison harness. I kept the **backend** MediaPipe pipeline
alive alongside the browser one (just for debugging), with a separate
hook that continued to capture JPEGs and ship them to `/ws/detect` to
extract landmarks server-side. Both hooks ran simultaneously, and the
overlay drew the **frontend MediaPipe in blue** and the **backend
MediaPipe in red** with identical projection math.

The blue and red skeletons were drawn from the same canvas, in the same
coordinate space, with the same `(1 - x) * canvas.width` formula. If the
projection math were the bug, both colours would be equally wrong. If
MediaPipe itself produced different landmarks in the two environments,
they'd diverge.

The red skeleton landed perfectly on the user's hand. The blue
skeleton was offset down-left.

That immediately ruled out projection, CSS, canvas sizing, and aspect
ratios. The bug had to be that **the two MediaPipe runs were producing
different landmark coordinates from the same source frame**.

## Why that's surprising

Both pipelines were using the **same MediaPipe Tasks API**, the **same
`.task` model file**, and the **same camera stream**. The Python and JS
ports of Tasks are the same C++ library underneath, just wrapped
differently. They should produce identical output.

But they don't.

## What's actually different

The Python backend's hand detector gets fed a base64 JPEG, which
`cv2.imdecode` turns into a `numpy.ndarray` at the JPEG's native size
(640×480 in our case). That gets handed to MediaPipe.

The browser code did this:

```ts
const handResult = detectors.hand.detect(video);  // <video> element
```

That's documented as a valid input — `HTMLVideoElement` is one of the
allowed `ImageSource` types. But here's the thing: **MediaPipe Tasks JS
samples the `<video>` element by uploading it as a texture at the
element's *display size***, not at the underlying stream's native
resolution.

In our app the `<video>` was rendered into a 527×394 box (CSS-scaled
from the 640×480 stream). So MediaPipe was getting a 527×394 input,
not 640×480. Different input → different model output → different
landmark coordinates.

The model still tracked the hand correctly within the downsampled
image, but the resulting normalised coordinates were in the
**527×394 space**, not the **640×480 space** that the user's actual
camera frame and the overlay canvas were aligned to. The "constant
offset" wasn't really constant — it was the aspect-and-scale
difference between two slightly different aspect ratios (1.333 vs
1.335) and slightly different pixel grids.

## The fix

Sample the `<video>` into an offscreen canvas at the **native stream
resolution** before passing it to MediaPipe:

```ts
const _sampleCanvas = document.createElement("canvas");
let _sampleCtx: CanvasRenderingContext2D | null = null;

function sampleVideoAtNativeRes(video: HTMLVideoElement): HTMLCanvasElement | null {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) return null;
  if (_sampleCanvas.width !== w) _sampleCanvas.width = w;
  if (_sampleCanvas.height !== h) _sampleCanvas.height = h;
  if (_sampleCtx === null) _sampleCtx = _sampleCanvas.getContext("2d");
  _sampleCtx!.drawImage(video, 0, 0, w, h);
  return _sampleCanvas;
}

export function detectFrame(detectors, video, _timestampMs) {
  const source = sampleVideoAtNativeRes(video) ?? video;
  const handResult = detectors.hand.detect(source);
  // ...
}
```

That's the entire fix. One offscreen canvas, sized to the video's
native resolution, redrawn each frame, passed to MediaPipe in place of
the `<video>` element. The blue skeleton snapped onto the red one
instantly.

## Why this is mildly infuriating

MediaPipe's documentation says any `ImageSource` works for `detect()`,
including `HTMLVideoElement`. It doesn't mention that passing the video
element samples it at display size rather than native size. The function
is permissive about its input but silently produces subtly different
output depending on which form you choose.

If you happened to size your video element to its native resolution
(e.g. `<video width="640" height="480">` with `object-fit` matching),
you'd never see this bug. The mistake we made was responsive CSS — the
video filled its containing box (`width: 100%; height: 100%`), which
meant the displayed size depended on layout. Different layout → different
input dimensions → different landmark coordinates.

Drawing through a sized canvas pins the input to a known resolution and
makes the pipeline behave identically to any other code path that
operates on a numpy array, an OpenCV `Mat`, or a JPEG buffer.

## The lesson

When two implementations of "the same model" produce different outputs,
the issue isn't usually in the model. It's in what you're feeding it.
Build a side-by-side diff harness early — even a temporary one — so you
can stop guessing about whether the bug is in projection or in input.
The harness was 60 lines of code and gave a definitive answer in 30
seconds, after I'd spent three hours bisecting CSS rules.
