# Vision mind

GLaDOS uses Gemma 4 E4B to describe what the camera sees and what changed
since the previous observed image. The Vision mind runs quietly in the
background; GLaDOS receives its latest scene, changes, and observation age.
Camera updates do not trigger unsolicited speech.
When people are visible, the scene describes their visible expressions, gaze
and posture before the surroundings. Obscured or small faces are described as
having unclear expressions, rather than assigning unseen feelings.
E4B returns these details in a dedicated `expressions` field, which is included
in **What I see**, the cached `vision_look` result, and camera context for replies.

## Setup

Start the multimodal E4B server described in [Gemma 4](gemma4.md), including its
`--mmproj` argument. The default endpoint is port 18080. Then run:

```bash
uv run glados webapp --config configs/glados_gemma4_llamacpp.yaml
```

Open http://127.0.0.1:8050/ and inspect **Minds → Vision**. The inspector shows
a live webcam stream with CPU face tracking, plus slower scene captions,
changes, measured update rate and frame timing.
The preview overlays the detected face box, gaze crosshair, and normalized X/Y
coordinates for that exact displayed frame. No face means no overlay. E4B
continues to receive the original image without annotations.
**Pause camera** releases the webcam and removes its context from conversation.
**Run once now** works while paused and releases the camera afterwards.
**Resume camera** restarts scheduled observations.

The front page shows the design's **Original** optic treatment beside the eye
when the camera is on: thermal colours, motion trails and contour lines. It
reuses `/api/vision/live?overlay=0`, so the CPU preview box is not burned into
this view. A circular **SUBJECT** target follows the eye's face-scanning gaze.
Motion heat fades by 95% in about 0.6 seconds. The feed keeps 80% of the source
in its limiting dimension, preserves its aspect ratio, and pans within the
image bounds. It follows the selected person
when faces are visible and slowly scans the room otherwise. Multiple people
receive alternating attention about every three seconds; position matching
keeps detector ordering changes from randomly switching the target.
Eye colours and eyelids drive the optic treatment. Each actual Vision HTTP
inference (background or question) emits a sequence cue that starts one short
blink in both the eye and feed. Queuing alone does not blink or mark the
user-facing eye as processing. Hiding the page releases its preview stream;
turning the camera off hides the optic feed and restores mouse gaze.

## Configuration

Add this section inside `Glados` in any configuration, including an API conversation profile:

```yaml
vision:
  enabled: true
  completion_url: "http://127.0.0.1:18080/v1/chat/completions"
  model: "gemma-4-E4B"
  camera_spec: 0
  interval_min_s: 2
  interval_max_s: 5
  frame_window_s: 0.25
  image_max_side: 512
  max_tokens: 256
  timeout_s: 10
  face_tracking: true
  face_backend: "yunet"
  face_interval_s: 0.03
  sleep_after_s: 5.0
```

Omit `vision`, or set `enabled: false`, to disable camera processing.
`camera_spec` accepts a camera index or OpenCV-supported stream URL.
`camera_index` is also accepted for existing camera configurations.
`completion_url`, `model` (the E4B server alias), and optional `api_key` belong
only to vision. They never inherit the conversation model, API URL, credentials,
or request options. If GLaDOS speaks using an API model, start E4B locally or
point vision at a remote E4B server. Webcam images go only to that endpoint;
the conversation API receives the resulting text.

## Frame selection and inference

A camera thread captures at up to 30 fps and keeps at most eight frames.
Numba scores a small grayscale image using Laplacian variance; higher scores
usually indicate sharper edges. Frame selection considers only the most recent
0.25-second window and rejects frames older than one second. It chooses the
sharpest candidate, breaking ties in favor of the newer frame.

The mind acquires capacity from the shared inference scheduler **before**
selecting a frame. A busy model therefore cannot create a backlog of old camera
requests. The schedule stays within the configured 2–5 second bounds. A small
96×72 grayscale CPU motion tracker measures changed pixels, compensates for
uniform exposure shifts, and smooths activity with fast attack and slower decay.
Still scenes favour the maximum delay; active scenes favour the minimum.
Each observation draws one random quantile. Scheduler polls reuse it, adjusting
the pending deadline to live motion instead of re-randomising the wait. Actual
frequency falls when inference takes longer or conversation needs capacity.
The inspector shows motion activity, current delay and completed frequency.

E4B receives up to four observations in one inference, oldest first with CURRENT
last. Each image is immediately preceded by its actual UTC capture timestamp,
seconds before CURRENT and the elapsed gap from the preceding observation.
The request also supplies the exact total span and adjacent gaps as explicit
metadata. Reports carry the host-computed `window_span_s`; generated recent-events
prose omits numeric durations so it does not replace those measurements.
The stable prompt asks for `scene` (what is happening now), `expressions`,
`changes` (since the immediately preceding image), and `recent_events` (the visible
sequence across the window). Intermittent snapshots must not be treated as
continuous video or evidence of unseen intervening actions. The 256-token output
budget accommodates all four fields.
Only successful observations advance the bounded four-image window; malformed
responses and timeouts do not. Changing cameras clears the window. The first
observation cannot establish recent events. Images and previews stay in bounded
process memory and are not written to conversation history or persistent mind
memory. The current scene, changes and recent-events report are included in
GLaDOS's visual context and in the Vision inspector. The `vision_look` tool reads
the latest observation and its actual age without another inference when called
without arguments.
For a specific detail, call `vision_look(question="Does my jacket have a zipper?")`.
This selects a fresh sharp frame after acquiring an interactive inference slot
and asks E4B to inspect that detail. The answer includes visible evidence and
the capture age; unclear or obscured details must be reported as uncertain.
Question inspections do not replace the background scene or its four-image
window. They use the same dedicated E4B endpoint for API conversation profiles.
A paused camera must be resumed before asking for a fresh inspection.
The latest question and answer are shown in **Minds → Vision**.
`question_image_max_side` defaults to 1024 (the native 640×480 camera image is
kept at that size); `question_max_tokens` defaults to 192.

This uses existing OpenCV, NumPy, Numba and requests dependencies. It does not
require additional Python packages. The 233 KB YuNet face model is bundled
with its MIT license; it runs through OpenCV on the CPU without GPU memory.

## Face tracking and sleep

By default, OpenCV's tiny YuNet neural face detector runs independently on the
CPU at up to 30 fps (`face_interval_s: 0.03`). It processes a
320-pixel-wide/long image and reports all detected faces. Actual frequency depends
on camera capture and CPU speed. A 30 ms detector interval lets each 30 fps
frame be tracked despite small capture timing variations. Local USB webcams
prefer MJPEG capture with two device buffers, allowing one frame to be queued
while the CPU processes the other. A single buffer halved this webcam's frame
rate; two buffers measured 29.8 fps with face tracking enabled and 28.5 fps
in the served live preview. E4B captions completed at 0.5 Hz. Results are recorded
in `docs/benchmarks/vision-face-yunet-2026-10-05.json`. On this machine, saved 640×480 webcam frames
with turned faces took about 3.3 ms per detection; the earlier Haar cascades
missed those faces. This small check is not a general accuracy benchmark.

The webcam inspector streams fresh JPEGs at up to 30 fps while open; the E4B
caption below updates on the motion-aware schedule (2–5 seconds by default). Face boxes, crosshairs and X/Y
labels come from detection on that exact displayed frame. JPEG encoding only
runs when a viewer requests the stream, and viewers share a cached encoding of
the same frame. Closing the inspector stops that viewer's stream. Detection
continues for avatar gaze and presence even when the inspector is closed.
E4B receives unannotated sharp frames, independent of the live preview.
Horizontal coordinates are reversed only when driving the eye; the preview and
its labels retain the original camera coordinates.

The avatar follows faces automatically when **Camera: ON**. When **Camera: OFF**,
it follows the mouse pointer. Missing or stale faces while the camera is on do
not switch gaze to the pointer. Face coordinates reach the eye through small
camera-only SSE updates at up to 30 Hz, separate from dashboard refreshes and
E4B captions. Idle glances cannot override a tracked face. Scanning alternates
approximate eye positions with shorter mouth glances, scaled to its bounding
box, and quiets down while following quick head movement. Blinking and
expressions remain. Nearness uses the larger of the normalized face width and
height: a face filling two-thirds of either image dimension gives full zoom.
The zoom ramps smoothly from ordinary size to 1.6× as a face approaches and
returns to ordinary size when camera tracking stops. Optional `face_backend: "haar"` uses OpenCV's bundled
frontal/profile cascades. Optional `face_backend: "e4b"` returns face presence
and Gemma's documented `face.box_2d: [y_min, x_min, y_max, x_max]` (0..1000) in
the same slower scene observation; see Google's
[object detection example](https://ai.google.dev/gemma/docs/capabilities/vision/image).
E4B boxes remain approximate and are drawn only on the selected observation
JPEG (`/api/vision/frame`), never on newer live webcam frames. The live preview
has no face overlay in E4B mode. Invalid or contradictory coordinates never
drive gaze or imply an empty room.

After five seconds of consecutive observations confirming no person, the idle eye closes into its sleeping
expression. A returning face, microphone speech, a user-facing inference, recent
user input, or keyboard/click interaction wakes it. Brief detector misses do not
cause sleep. A paused, disconnected, failed, or stale camera supplies unknown
presence and cannot put the eye to sleep. Sleep changes only the avatar; camera
capture, vision observations, and microphone input continue so she can wake.
An uncertain result, failed observation, or stale gap restarts absence timing.
All face-detection backends are approximate; this signal is only
used for animation. Set `face_tracking: false` to disable it.

`image_max_side` already downsizes the background images before transmission.
The default remains 512 pixels: in local tests, reducing it to 384 reduced two-image
prompt token counts by 80 but made the face boxes drift more. At 256, the local
llama.cpp preprocessor's minimum image size meant no further token reduction.
Text generation dominated the remaining latency. Detailed questions independently
retain the native camera resolution through `question_image_max_side`.

Local E4B checks on 2026-10-05 found the turned face in three test images that
Haar missed, and correctly reported no person in a shelf-only crop. Face boxes
remained approximate. Tests of the combined previous/current-image prompt took
about 1.5–2.5 seconds while the live vision mind was also running. These are a
small practical check, not a general detector accuracy benchmark. Results are
recorded in `docs/benchmarks/vision-face-e4b-2026-10-05.json`; camera images are
not stored there.
