# CameraDrive AI 0.6.0 — Low-Light Articulated Pedals

CameraDrive AI is a Windows video-game accessibility controller. A webcam tracks small, comfortable foot articulation and sends Xbox-style trigger input:

- Right foot: accelerator / right trigger
- Left foot: brake / left trigger
- Steering: intentionally centred in pedals-only mode

The webcam estimates **pedal demand/travel**, not physical pressure or force. Measuring real pressure requires an instrumented pedal such as a load cell or Hall-effect sensor. This program is only for video games—never a real vehicle, powered mobility device, or machinery.

## Install and calibrate

1. Install 64-bit Python 3.11.
2. Run `install.bat`.
3. Put the camera and lighting in their normal playing positions, then run `run_camera_fps_test_and_apply.bat`.
4. Run `run_preview_max_leg_accuracy.bat`.
5. Press **F9** and calibrate with relaxed, very small foot movements.
6. When tracking is stable, run `install_gamepad_backend.bat` once and restart Windows after its driver installation.
7. Run `run_max_leg_accuracy_windowed.bat`, press **F8**, and test the triggers with `joy.cpl`.
8. Use `run_max_leg_accuracy.bat` for the transparent in-game overlay.

The first tracking launch downloads the selected MediaPipe pose model into `models/`. The virtual controller exists only while a non-preview CameraDrive launcher is running.

## What changed in 0.6.0

### Low-light capture

- DirectShow auto exposure is allowed to settle and is then locked when the camera supports it.
- Optional manual exposure, gain, brightness, autofocus, and focus values are available in `config.json`.
- A fixed gamma transform and mild, bounded CLAHE enhancement are applied once after resizing and shared by MediaPipe and optical flow.
- The camera benchmark now balances real FPS with full-frame brightness, darkness, texture corners, sharpness, frame cadence, and duplicate frames.
- The HUD adds `LIGHT` and `DARK` readings. Very low `LIGHT` or high `DARK` means the camera needs more light or a lower FPS mode.

The gamma and CLAHE settings stay fixed, and both tracking paths receive the same enhanced pixels. Frame-varying denoising or automatic gamma can smear micro-movement and break optical-flow brightness consistency.

### Angle-tolerant foot articulation

Each foot is now described in a coordinate system attached to its own shin. Translation, uniform scale, and in-plane rotation of the whole leg therefore do not become false pedal input. The model combines:

The camera aspect ratio is corrected before geometry is calculated, so rotating a leg inside a non-square 4:3 or 16:9 image does not distort the inferred articulation.

- Foot direction relative to the shin
- Toe travel relative to the ankle
- Optional heel and sole direction/length
- Optional thigh-to-shin articulation
- Optional low-weight 3-D cues
- Nine camera-rate texture samples around every tracked body point

Knee, ankle, and toe form the required core. Missing or geometrically implausible heel/hip values and missing 3-D values are explicitly masked instead of being interpreted as movement. Real landmark confidence and person-mask support are kept separate from the HUD's point-source markers, so weak bedding anchors cannot masquerade as strong foot points.

The trigger curve now preserves the complete range. A tiny comfortable calibrated movement can still reach 100%, but intermediate foot positions remain distinct instead of reaching full throttle in the first third of travel.

## Calibration without strain

Press **F9**, then hold:

1. Both feet relaxed and still.
2. The smallest comfortable, repeatable right-foot accelerator action.
3. The smallest comfortable, repeatable left-foot brake action.

That tiny held action becomes full trigger travel. Do not force a larger movement. Keep hips or upper thighs, knees, ankles, heels, and toe areas visible when possible. Keep blanket folds away from the foot edges.

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
- **Esc**: neutralize output and exit

HUD diagnostics:

- **CAM**: webcam delivery rate
- **AI**: semantic pose detections per second
- **TRACK**: high-rate motion updates
- **LIGHT**: median frame brightness, from 0 to 255
- **DARK**: percentage of very dark frame pixels

## Launchers

- `run_preview_max_leg_accuracy.bat`: calibration and preview; no controller output
- `run_max_leg_accuracy_windowed.bat`: controller output with a diagnostic window
- `run_max_leg_accuracy.bat`: controller output with a transparent overlay
- `run_camera_fps_test_and_apply.bat`: select and save the best usable camera mode
- `run_tests.bat`: run deterministic low-light, camera-control, mapping, migration, and benchmark tests

## Troubleshooting

- If a point jumps, add light, reduce blur, move blanket edges away, and keep the leg chain visible.
- If the trigger flickers at neutral, recalibrate with still bedding and camera mounting before raising dead zones.
- If `TRACK` is far below `CAM`, hide the preview with **F10** and close background-heavy programs.
- If no controller appears, run `install_gamepad_backend.bat`, complete its driver installer, restart Windows, and retry.
- If the controller works in `joy.cpl` but not the game, close competing controller emulators and temporarily disconnect other gamepads.

See `MANIFEST.md` for the repository contents.
