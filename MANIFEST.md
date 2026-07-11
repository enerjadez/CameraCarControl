# CameraDrive AI 0.5.0 clean-package manifest

This package contains only the max-leg-accuracy pedals workflow and its required setup files.

| File | Purpose |
|---|---|
| `camera_drive.py` | Main camera, AI tracking, calibration, HUD, and virtual-controller application |
| `camera_fps_test.py` | Webcam mode benchmark and configuration helper |
| `config.json` | Default v0.5.0 max-leg-accuracy configuration |
| `install.bat` | Creates the Python 3.11 virtual environment and installs core dependencies |
| `install_gamepad_backend.bat` | Installs the optional virtual Xbox-controller backend |
| `requirements.txt` | Core Python dependency list |
| `requirements-gamepad.txt` | Virtual-controller dependency list |
| `run_preview_max_leg_accuracy.bat` | Max-accuracy calibration and preview, without game output |
| `run_max_leg_accuracy_windowed.bat` | Max-accuracy game output with a diagnostic window |
| `run_max_leg_accuracy.bat` | Max-accuracy game output with a transparent overlay |
| `run_camera_fps_test_and_apply.bat` | Measures webcam modes and applies the fastest reliable result |
| `README.md` | Setup, calibration, game-use, and troubleshooting guide |
| `LICENSE` | Software license |
| `VERSION.txt` | Package version (`0.5.0`) |
| `MANIFEST.md` | This file list |

Generated after installation and deliberately not bundled: `.venv/`, `models/`, configuration backups, and caches.

Deliberately excluded: v0.1–v0.4 packages, upgrade overlays/notes, legacy and duplicate profile launchers, duplicate source trees, checksums, changelog, and test report.
