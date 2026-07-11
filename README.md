# CameraDrive AI 0.5.0 — Max Leg Accuracy

CameraDrive AI is a Windows video-game accessibility controller. A webcam tracks small, comfortable foot actions and sends Xbox-style trigger input:

- Right foot: accelerator / right trigger
- Left foot: brake / left trigger
- Steering: intentionally centred in this pedals-only package

It is for video games only. Never connect or adapt it to a real vehicle, powered mobility device, or machinery.

## Install

1. Install 64-bit Python 3.11.
2. Run `install.bat` to create `.venv` and install camera-tracking dependencies.
3. Optionally run `run_camera_fps_test_and_apply.bat` to measure and save the webcam's fastest reliable mode.
4. Run `run_preview_max_leg_accuracy.bat`.
5. Press **F9** and calibrate with relaxed, very small foot movements.
6. When tracking is stable, run `install_gamepad_backend.bat` once and restart Windows after its driver installation.
7. Run `run_max_leg_accuracy_windowed.bat`, press **F8**, and test the virtual triggers with `joy.cpl`.
8. Use `run_max_leg_accuracy.bat` for the transparent in-game overlay.

The first tracking launch downloads the selected MediaPipe pose model into a local `models` folder, so an internet connection is needed once. The virtual controller exists only while a non-preview CameraDrive launcher is running.

## Launchers

- `run_preview_max_leg_accuracy.bat`: calibration and tracking preview; no virtual controller
- `run_max_leg_accuracy_windowed.bat`: virtual controller with a normal diagnostic window
- `run_max_leg_accuracy.bat`: virtual controller with a transparent overlay
- `run_camera_fps_test_and_apply.bat`: measure modes and save the fastest reliable result to `config.json`

## Calibration without strain

Press **F9**, then hold each pose while the capture bar fills:

1. Both feet relaxed and still.
2. The smallest comfortable, repeatable right-foot accelerator action.
3. The smallest comfortable, repeatable left-foot brake action.

Do not force a larger movement. If calibration cannot separate the action from camera noise, move the camera closer, improve lighting, uncover the leg and foot edges, or steady the bedding and camera mount.

For the strongest leg lock, keep both upper thighs or hips, knees, ankles, heels, and toe areas visible. Keep the legs visually separated and blanket folds away from the ankles and toes.

## Controls and status

- **F8**: enable or pause game output
- **F9**: restart calibration
- **F10**: show or hide the camera preview
- **Esc**: neutralize output and exit

HUD rates:

- **CAM**: frames delivered by the webcam
- **AI**: semantic pose detections per second
- **TRACK**: high-rate control-tracking updates

`AI` can be lower than `CAM` and `TRACK`; fast optical-flow tracking continues between AI anchors. Trust the measured `CAM` rate rather than the FPS requested in `config.json`.

## Game setup

1. Start `run_max_leg_accuracy_windowed.bat` before the game.
2. Complete calibration and press **F8** until the status reads `ACTIVE`.
3. Press Windows+R, enter `joy.cpl`, and confirm the right and left virtual triggers respond.
4. Leave CameraDrive running, start the game, and select its normal Xbox-controller profile.
5. After testing, switch to `run_max_leg_accuracy.bat` if you want the overlay.

Use borderless or windowed-fullscreen mode if an exclusive fullscreen game hides the overlay.

## Troubleshooting

- If a leg point jumps, add even light, reduce motion blur, move blanket edges away, and keep the full hip-to-toe chain visible.
- If `TRACK` is far below `CAM`, close other camera users and background-heavy applications, then retest the webcam mode.
- If no virtual controller appears, run `install_gamepad_backend.bat`, complete its driver installer, restart Windows, and retry.
- If the game ignores a controller that works in `joy.cpl`, close other controller emulators and temporarily disconnect competing gamepads while testing.

See `MANIFEST.md` for the exact clean-package contents.
