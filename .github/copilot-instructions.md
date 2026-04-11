# Project Guidelines

## General Principles
1. In any process, task, or dialogue, user feedback must be requested using the askQuestions tool after each stage is completed.
2. Adjust behavior based on the received user feedback.
3. A process is considered complete only when the user explicitly indicates "end" or "no further interaction needed."
4. All steps must be repeated unless an end instruction is received.
5. Before completing a task, the askQuestions tool must be used to prompt the user for feedback.
6. Brief: concise, repo-specific guidance so an AI agent can be productive quickly.

## Code Style
- Use Python type hints and pathlib-based path handling, following patterns in train.py, predict.py, and web/yolo_utils.py.
- Keep CLI files (train.py, predict.py, app.py) focused on argument parsing and orchestration.
- Put reusable YOLO and path logic in web/yolo_utils.py instead of duplicating logic in multiple entry points.
- Keep root compatibility wrapper app.py thin and import-forwarding only.

## Architecture
- Primary runtime modules:
  - train.py: training entry point.
  - predict.py: batch image inference entry point.
  - web/app.py: Flask upload + result UI.
  - web/yolo_utils.py: shared YOLO load/train/predict/path utilities.
- Dataset flow: data.yaml -> prepare_data_yaml() -> data.local.yaml with resolved absolute split paths.
- Weights resolution: prefer runs/fruit_detector/weights/best.pt, then newest best.pt under runs/.
- Web file flow: uploads are saved under web/uploads and rendered outputs under web/results.

## Build and Test
- Create environment and install dependencies:
  - python -m venv .venv
  - PowerShell: .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt
  - For web UI work, install Flask if missing: pip install flask
- Common run commands:
  - python train.py
  - python train.py --dataset dataset2
  - python train.py --data-yaml dataset2/data.yaml
  - python predict.py --image test_image/your_image.jpg
  - python app.py
- There is no formal automated test suite yet. Validate changes with:
  - one quick train or weight-load check,
  - one predict run,
  - one web upload round-trip.
  - lightweight pre-checks: python train.py --help and python predict.py --help

## Conventions
- Preserve dataset selection behavior in train.py (--dataset and --data-yaml).
- When editing dataset handling, keep prepare_data_yaml() fallback behavior for split paths intact.
- Handle missing model weights clearly (predict/web should report that training is required).
- Prefer Windows-friendly command examples in user-facing updates.
- Keep model defaults consistent with code (train.py defaults to yolo11x.pt unless explicitly changed).
- Keep web-facing errors actionable; avoid exposing raw stack traces where practical.

## Environment Notes
- Training device is auto-selected by get_training_device() (CUDA when available, otherwise CPU).
- Weights lookup order must remain: runs/fruit_detector/weights/best.pt, then newest best.pt under runs/.
- Web paths are runtime-created under web/uploads and web/results; do not hardcode alternate output locations without updating web/yolo_utils.py and web/app.py.

## Documentation Links
- Use links instead of duplicating documentation details:
  - README.md (setup and usage)
  - reference/TASK1-3_說明.md (project background)