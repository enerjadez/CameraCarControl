# CameraDrive AI 0.6.0 manifest

| File | Purpose |
|---|---|
| `camera_drive.py` | Main AI tracking, articulated-foot calibration/mapping, HUD, and controller application |
| `camera_imaging.py` | Shared camera controls, deterministic low-light enhancement, and light metrics |
| `camera_fps_test.py` | Quality-aware webcam benchmark and configuration helper |
| `config.json` | Default v0.6.0 low-light articulated-pedal configuration |
| `install.bat` | Creates the Python 3.11 environment and installs core dependencies |
| `install_gamepad_backend.bat` | Installs the optional virtual Xbox-controller backend |
| `requirements.txt` | Core Python dependencies |
| `requirements-gamepad.txt` | Virtual-controller dependency |
| `run_preview_max_leg_accuracy.bat` | Max-accuracy calibration and preview without controller output |
| `run_max_leg_accuracy_windowed.bat` | Max-accuracy output with a diagnostic window |
| `run_max_leg_accuracy.bat` | Max-accuracy output with a transparent overlay |
| `run_camera_fps_test_and_apply.bat` | Selects and applies the best usable webcam mode |
| `run_tests.bat` | Runs the synthetic regression suite |
| `tests/test_v06.py` | Low-light, camera-control, angle-invariance, mapping, migration, and benchmark tests |
| `tests/config_v05_fixture.json` | Original v0.5 configuration fixture used to verify safe, idempotent migration |
| `README.md` | Setup, calibration, configuration, game-use, and troubleshooting guide |
| `LICENSE` | MIT license |
| `VERSION.txt` | Application version (`0.6.0`) |
| `.gitignore` | Excludes generated environments, models, reports, backups, and caches |

Generated and deliberately untracked: `.venv/`, `models/`, `__pycache__/`, camera reports, configuration backups, and temporary files.
