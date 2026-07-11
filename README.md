# CameraDrive AI 0.7.0 — Fixed-Heel Triangle Pedals

CameraDrive AI is a Windows video-game accessibility controller. You select three fixed tracking points on each foot, then a webcam follows heel-pivot movement and sends Xbox-style trigger input:

- Right foot: accelerator / right trigger
- Left foot: brake / left trigger
- Steering: intentionally centred in pedals-only mode

The webcam estimates **pedal demand/travel**, not physical pressure or force. Measuring real pressure requires an instrumented pedal such as a load cell or Hall-effect sensor. This program is only for video games—never a real vehicle, powered mobility device, or machinery.

## Install and calibrate

Extract the complete ZIP into a writable folder such as Documents; do not run the batch files from inside the compressed-file preview. Internet access is required during installation and the first pose-model download.

1. Install 64-bit Python 3.11 with the Windows `py` launcher.
2. Run `install.bat`.
3. Put the camera and lighting in their normal playing positions, then run `run_camera_fps_test_and_apply.bat`.
4. Run `run_preview_max_leg_accuracy.bat`. CameraDrive freezes one setup image so every click refers to the same frame. Keep both feet still and click the six requested points in order: **right heel, right ankle, right toe, left heel, left ankle, left toe**.
5. If a click is misplaced, press **F7** to clear all six points and restart anchor setup. After the triangles lock, follow the neutral, accelerator, and brake calibration prompts using relaxed, very small heel tilts.
6. When tracking is stable, run `install_gamepad_backend.bat`. It installs the Python backend, launches the bundled x64 virtual-controller driver with a Windows administrator prompt when needed, and verifies that a controller can be created. Restart Windows if prompted.
7. Run `run_fixed_heel_pedals_windowed.bat`, set the six anchors again, complete calibration, press **F8**, and test the triggers with `joy.cpl`.
8. Leave that window open while playing.

The six anchors are deliberately **session-only**: set them again whenever CameraDrive starts or the camera or your body moves. Saving screen coordinates would risk attaching a later session to bedding or empty space. The first tracking launch downloads the selected MediaPipe pose model into `models/`. The virtual controller exists only while a non-preview CameraDrive launcher is running.

## What changed in 0.7.0

### User-set hard-lock triangles

- Each foot uses exactly three user-selected points: heel, ankle, and toe.
- The heel is the pivot. Calibration learns the small neutral-to-pressed triangle tilt that is comfortable for you.
- Pedal output comes from the triangle's heel-relative rotation and shape, not whole-leg or whole-foot translation. Moving the complete triangle together does not become false throttle or brake.
- MediaPipe still identifies and diagnoses the legs, but it cannot pull a manually selected triangle to a different semantic point or a blanket edge.
- Each selected point is carried by a 5×5 cloud of 25 camera-rate motion samples with forward/backward validation. Every frame is rechecked against the original clicked texture, and the triangle is bounded to its confirmed edge lengths and orientation, so slow coherent drift cannot accumulate onto bedding.
- If a required hard-lock point is lost, that pedal returns to zero instead of freezing or silently snapping somewhere else. Press **F7** to set the anchors again.

The webcam still infers pedal travel from motion; it cannot measure literal physical pressure. A load cell or another pedal sensor is required for real force measurement.

### Low-light capture

- DirectShow auto exposure is allowed to settle and is then locked when the camera supports it.
- Optional manual exposure, gain, brightness, autofocus, and focus values are available in `config.json`.
- A fixed gamma transform and mild, bounded CLAHE enhancement are applied once after resizing and shared by MediaPipe and optical flow.
- The camera benchmark now balances real FPS with full-frame brightness, darkness, texture corners, sharpness, frame cadence, and duplicate frames.
- The HUD adds `LIGHT` and `DARK` readings. Very low `LIGHT` or high `DARK` means the camera needs more light or a lower FPS mode.

The gamma and CLAHE settings stay fixed, and both tracking paths receive the same enhanced pixels. Frame-varying denoising or automatic gamma can smear micro-movement and break optical-flow brightness consistency.

### Fine heel-tilt mapping

The camera aspect ratio is corrected before triangle geometry is calculated. The mapping tolerates any fixed camera angle used for both neutral and pressed calibration: it compares the current heel-pivot triangle with the endpoints learned in that same view. A small comfortable tilt can reach 100%, while intermediate foot angles remain distinct across the complete trigger range. If the camera or foot position changes materially during play, press **F7** and anchor again; a moving camera and a rotating foot are not distinguishable from one webcam view.

## Calibration without strain

First set the six anchor points in the requested order. Press **F7** at any time to discard them and start again. Once both triangles show locked, follow the calibration prompts and hold:

1. Both feet relaxed and still.
2. The smallest comfortable, repeatable right-foot accelerator action.
3. The smallest comfortable, repeatable left-foot brake action.

That tiny held heel tilt becomes full trigger travel. The default minimum useful tilt is only 2 degrees. Do not force a larger movement. Keep each heel, ankle, and toe visible, light the foot surfaces evenly, and keep blanket folds away from all six selected points.

## Low-light configuration

Recommended defaults:

```json
"camera_exposure_mode": "auto_lock",
"low_light_enhancement_enabled": true,
"low_light_gamma": 0.72,
"low_light_clahe_clip_limit": 1.6
```

Exposure modes are `unchanged`, `auto`, `auto_lock`, or `manual`. Camera property support and ranges vary. If auto-lock is unsupported, CameraDrive leaves the driver in control and continues safely.

If the image remains dark, run the benchmark again. A clean 50 or 60 FPS mode can track better than noisy or duplicated 120 FPS. Additional diffuse light is more effective than aggressive software brightening.

## Controls and status

- **F8**: enable or pause game output
- **F9**: restart calibration
- **F10**: show or hide the camera preview
- **F7**: clear and restart the six-point anchor setup
- **Esc**: neutralize output and exit

HUD diagnostics:

- **CAM**: webcam delivery rate
- **AI**: semantic pose detections per second
- **TRACK**: high-rate motion updates
- **LIGHT**: median frame brightness, from 0 to 255
- **DARK**: percentage of very dark frame pixels

## Launchers

- `run_fixed_heel_pedals_windowed.bat`: recommended six-point fixed-heel setup with controller output
- `run_preview_max_leg_accuracy.bat`: clickable six-point setup and calibration preview; no controller output
- `run_camera_fps_test_and_apply.bat`: select and save the best usable camera mode

The source repository also retains `run_max_leg_accuracy_windowed.bat` and `run_max_leg_accuracy.bat` as old-name compatibility launchers, plus `run_tests.bat` and the deterministic regression suite. They are intentionally omitted from the clean runtime ZIP because they are not needed to install or play.

## Troubleshooting

- If an anchor was placed incorrectly or a triangle unlocks, press **F7**, keep the feet still, and select all six points again.
- If setup restarts with a low-texture warning, click a more distinct skin, sock, or shoe feature at the requested joint rather than a smooth shadow.
- If a point slips, add light, reduce blur, move blanket edges away, and choose a textured spot on the requested heel, ankle, or toe area.
- If the trigger flickers at neutral, recalibrate with still bedding and camera mounting before raising dead zones.
- If `TRACK` is far below `CAM`, hide the preview with **F10** and close background-heavy programs.
- If no controller appears, run `install_gamepad_backend.bat`, complete its driver installer, restart Windows, and retry.
- If the controller works in `joy.cpl` but not the game, close competing controller emulators and temporarily disconnect other gamepads.

See `MANIFEST.md` for the clean ZIP contents and repository-only development files.
