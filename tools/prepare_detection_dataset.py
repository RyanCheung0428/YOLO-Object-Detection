from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = (("train", 0.8), ("valid", 0.1), ("test", 0.1))
SEED = 42


def list_classes(dataset_dir: Path) -> list[Path]:
    # Treat non-split subdirectories as source class folders.
    ignored = {"train", "valid", "test"}
    return sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_dir() and path.name not in ignored
        ],
        key=lambda path: path.name.lower(),
    )


def gather_images(class_dirs: list[Path]) -> list[tuple[Path, int, str]]:
    # Build (image path, class id, class name) tuples from class subtrees.
    samples: list[tuple[Path, int, str]] = []
    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, class_id, class_name))
    return samples


def prepare_output_dirs(dataset_dir: Path) -> None:
    # Rebuild split folders to avoid mixing stale and new prepared files.
    for split, _ in SPLITS:
        split_dir = dataset_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split, _ in SPLITS:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def choose_split(index: int, total: int) -> str:
    # Deterministic ratio split after shuffle for reproducible dataset layout.
    train_cutoff = int(total * 0.8)
    valid_cutoff = train_cutoff + int(total * 0.1)
    if index < train_cutoff:
        return "train"
    if index < valid_cutoff:
        return "valid"
    return "test"


def safe_filename(class_name: str, stem: str, index: int, suffix: str) -> str:
    # Normalize names so generated paths stay shell-friendly across platforms.
    safe_class = class_name.replace(" ", "_")
    safe_stem = stem.replace(" ", "_")
    return f"{safe_class}_{safe_stem}_{index:06d}{suffix.lower()}"


def link_or_copy(src: Path, dst: Path) -> None:
    # Prefer hard links for speed/storage, fallback to copy when unavailable.
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_label(label_path: Path, class_id: int) -> None:
    if label_path.exists():
        return
    # Fallback pseudo-box for converting classification folders to detection format.
    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def write_data_yaml(dataset_dir: Path, class_dirs: list[Path]) -> None:
    # Emit YOLO data.yaml with relative split paths and class names.
    names = [path.name for path in class_dirs]
    lines = [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend([f"  - {name}" for name in names])
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_source_dir(project_root: Path, source_arg: str) -> Path:
    """Resolve source path and auto-detect common dataset folders for default input."""
    source_dir = Path(source_arg)
    if source_dir.is_absolute():
        return source_dir

    candidate = (project_root / source_dir).resolve()
    if candidate.exists():
        # For default mode, ignore stale output folders that only contain split dirs.
        if source_arg != "dataset" or list_classes(candidate):
            return candidate

    if source_arg != "dataset":
        return candidate

    # Backward-compatible auto-detection for common repository layouts.
    fallback_candidates = [
        project_root / "dataset-Fruits-262" / "Fruit-262",
        project_root / "dataset" / "Fruit-262",
        project_root / "dataset",
    ]
    for path in fallback_candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    return candidate


def main() -> None:
    # Keep CLI logic minimal; heavy lifting stays in helper functions above.
    parser = argparse.ArgumentParser(
        description="Convert class-folder images to YOLO detection format with pseudo labels."
    )
    parser.add_argument(
        "--source-dir",
        default="dataset",
        help=(
            "Folder that directly contains class subfolders. "
            "Example: dataset-Fruits-262/Fruit-262"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_fruit262",
        help=(
            "Prepared output folder for train/valid/test and data.yaml. "
            "Example: dataset_fruit262"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_dir = resolve_source_dir(project_root, args.source_dir)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = list_classes(source_dir)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {source_dir}")

    samples = gather_images(class_dirs)
    if not samples:
        raise RuntimeError(f"No images found under class directories in: {source_dir}")

    random.Random(SEED).shuffle(samples)
    prepare_output_dirs(output_dir)

    split_counts = {"train": 0, "valid": 0, "test": 0}
    for index, (image_path, class_id, class_name) in enumerate(samples):
        split = choose_split(index, len(samples))
        filename = safe_filename(class_name, image_path.stem, index, image_path.suffix)

        image_dst = output_dir / split / "images" / filename
        label_dst = output_dir / split / "labels" / f"{Path(filename).stem}.txt"

        link_or_copy(image_path, image_dst)
        write_label(label_dst, class_id)
        split_counts[split] += 1

    write_data_yaml(output_dir, class_dirs)

    print("Conversion finished.")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(samples)}")
    print(f"Train images: {split_counts['train']}")
    print(f"Valid images: {split_counts['valid']}")
    print(f"Test images: {split_counts['test']}")
    print(f"Updated data.yaml: {(output_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
