# CameraDrive AI 0.7.0 manifest

## Clean Windows ZIP

| File | Purpose |
|---|---|
| `camera_drive.py` | Main low-light tracking, user-set hard-lock triangles, heel-pivot mapping, HUD, and controller application |
| `camera_imaging.py` | Shared camera controls, deterministic low-light enhancement, and light metrics |
| `camera_fps_test.py` | Quality-aware webcam benchmark and configuration helper |
| `config.json` | Default v0.7.0 low-light fixed-heel triangle configuration |
| `install.bat` | Creates the Python 3.11 environment and installs core dependencies |
| `install_gamepad_backend.bat` | Installs the optional virtual Xbox-controller backend |
| `requirements.txt` | Core Python dependencies |
| `requirements-gamepad.txt` | Virtual-controller dependency |
| `run_fixed_heel_pedals_windowed.bat` | Recommended clickable six-anchor fixed-heel setup with controller output |
| `run_preview_max_leg_accuracy.bat` | Clickable six-anchor setup and preview without controller output |
| `run_camera_fps_test_and_apply.bat` | Selects and applies the best usable webcam mode |
| `README.md` | Six-anchor setup, calibration, low-light configuration, game-use, and troubleshooting guide |
| `MANIFEST.md` | This exact clean-package and repository inventory |
| `LICENSE` | MIT license |
| `VERSION.txt` | Application version (`0.7.0`) |

## Repository-only development and compatibility files

| File | Purpose |
|---|---|
| `run_max_leg_accuracy_windowed.bat` | Old-name compatibility launcher for fixed-heel controller output |
| `run_max_leg_accuracy.bat` | Second old-name compatibility launcher |
| `run_tests.bat` | Runs the deterministic regression suite |
| `tests/test_v06.py` | Low-light, camera-control, mapping, migration, and benchmark tests |
| `tests/test_v07.py` | Frozen-frame clicks, canonical texture lock, triangle mapping, drift rejection, and immediate-neutral tests |
| `tests/config_v05_fixture.json` | Legacy configuration migration fixture |
| `.gitignore` | Excludes generated environments, models, reports, backups, and caches |

Generated and deliberately untracked: `.venv/`, `models/`, `__pycache__/`, camera reports, configuration backups, and temporary files.
