import argparse
import csv
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 448
DEFAULT_REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
DEFAULT_MIRROR = "https://hf-mirror.com"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_BLACKLIST = PROJECT_ROOT / "meta_tag_black_list.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPECIAL_TAGS = {
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple_boys",
    "male_focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple_others",
}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def parse_tag_list(tags_arg: str | None) -> list[str]:
    if not tags_arg:
        return []
    return list(dict.fromkeys([tag.strip() for tag in tags_arg.split(",") if tag.strip()]))


def dedupe_preserve_order(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tags))


def list_images(input_dir: Path) -> list[Path]:
    return sorted([path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    width, height = image.size
    size = max(width, height)
    squared = Image.new("RGB", (size, size), (255, 255, 255))
    squared.paste(image, ((size - width) // 2, (size - height) // 2))
    squared = squared.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(squared, dtype=np.float32)
    return array[:, :, ::-1]


def load_blacklist(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}


def download_model(repo_id: str, model_dir: Path, force_download: bool) -> tuple[Path, Path, bool]:
    tokens = repo_id.split("/")
    if len(tokens) > 2:
        base_repo_id = "/".join(tokens[:2])
        subdir = "/".join(tokens[2:])
        model_location = model_dir / base_repo_id.replace("/", "_") / subdir
        onnx_model_name = "model_optimized.onnx"
        default_format = False
    else:
        base_repo_id = repo_id
        subdir = None
        model_location = model_dir / repo_id.replace("/", "_")
        onnx_model_name = "model.onnx"
        default_format = True

    model_location.mkdir(parents=True, exist_ok=True)

    if subdir is None:
        files = ["selected_tags.csv", "model.onnx"]
        for file_name in files:
            target = model_location / file_name
            if force_download or not target.exists():
                hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=str(model_location), force_download=force_download)
    else:
        files = [onnx_model_name, "tag_mapping.json"]
        for file_name in files:
            target = model_location / file_name
            if force_download or not target.exists():
                hf_hub_download(
                    repo_id=base_repo_id,
                    filename=file_name,
                    subfolder=subdir,
                    local_dir=str(model_dir / base_repo_id.replace("/", "_")),
                    force_download=force_download,
                )

    return model_location, model_location / onnx_model_name, default_format


def load_tag_metadata(model_location: Path, default_format: bool, remove_underscore: bool) -> dict:
    def normalize_tag(tag: str) -> str:
        return tag.replace("_", " ") if remove_underscore and len(tag) >= 4 else tag

    if default_format:
        with open(model_location / "selected_tags.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0]
        assert header[:3] == ["tag_id", "name", "category"], f"unexpected csv format: {header}"
        data_rows = rows[1:]
        return {
            "default_format": True,
            "rating_tags": [normalize_tag(row[1]) for row in data_rows if row[2] == "9"],
            "general_tags": [normalize_tag(row[1]) for row in data_rows if row[2] == "0"],
            "character_tags": [normalize_tag(row[1]) for row in data_rows if row[2] == "4"],
        }

    with open(model_location / "tag_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    tag_id_to_tag = {}
    tag_id_to_category = {}
    for tag_id, info in mapping.items():
        tag_id_to_tag[int(tag_id)] = normalize_tag(info["tag"])
        tag_id_to_category[int(tag_id)] = info["category"]

    return {
        "default_format": False,
        "tag_id_to_tag": tag_id_to_tag,
        "tag_id_to_category": tag_id_to_category,
    }


def make_session(model_path: Path) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(model_path), providers=providers)


def categorize_default_format(
    prob: np.ndarray,
    metadata: dict,
    thresholds: dict[str, float],
    include_quality: bool,
    include_rating: bool,
    undesired_tags: set[str],
) -> dict[str, list[str]]:
    categories = {key: [] for key in ["special", "character", "copyright", "artist", "general", "meta", "rating", "quality"]}
    general_tags = metadata["general_tags"]
    character_tags = metadata["character_tags"]
    rating_tags = metadata["rating_tags"]

    for index, p in enumerate(prob[4:]):
        if index < len(general_tags) and p >= thresholds["general"]:
            tag = general_tags[index]
            if tag in undesired_tags:
                continue
            if tag in SPECIAL_TAGS:
                categories["special"].append(tag)
            else:
                categories["general"].append(tag)
        elif index >= len(general_tags) and p >= thresholds["character"]:
            tag = character_tags[index - len(general_tags)]
            if tag not in undesired_tags:
                categories["character"].append(tag)

    if include_rating:
        rating_index = int(prob[:4].argmax())
        rating_tag = rating_tags[rating_index]
        if rating_tag not in undesired_tags:
            categories["rating"].append(rating_tag)

    if include_quality:
        categories["quality"] = []

    return categories


def categorize_extended_format(
    prob: np.ndarray,
    metadata: dict,
    thresholds: dict[str, float],
    include_quality: bool,
    include_rating: bool,
    undesired_tags: set[str],
) -> dict[str, list[str]]:
    categories = {key: [] for key in ["special", "character", "copyright", "artist", "general", "meta", "rating", "quality"]}
    prob = 1.0 / (1.0 + np.exp(-prob))

    best_rating = None
    best_rating_score = -1.0
    best_quality = None
    best_quality_score = -1.0

    min_threshold = min(thresholds.values())
    for index in np.where(prob >= min_threshold)[0]:
        if index not in metadata["tag_id_to_tag"]:
            continue
        tag = metadata["tag_id_to_tag"][int(index)]
        category = metadata["tag_id_to_category"][int(index)]
        score = float(prob[int(index)])

        if tag in undesired_tags:
            continue

        if category == "Rating":
            if score > best_rating_score:
                best_rating = tag
                best_rating_score = score
            continue
        if category == "Quality":
            if score > best_quality_score:
                best_quality = tag
                best_quality_score = score
            continue

        if category == "General" and score >= thresholds["general"]:
            if tag in SPECIAL_TAGS:
                categories["special"].append(tag)
            else:
                categories["general"].append(tag)
        elif category == "Character" and score >= thresholds["character"]:
            categories["character"].append(tag)
        elif category == "Copyright" and score >= thresholds["copyright"]:
            categories["copyright"].append(tag)
        elif category == "Artist" and score >= thresholds["artist"]:
            categories["artist"].append(tag)
        elif category in {"Meta", "Model"} and score >= thresholds["meta"]:
            categories["meta"].append(tag)

    if include_rating and best_rating is not None:
        categories["rating"].append(best_rating)
    if include_quality and best_quality is not None:
        categories["quality"].append(best_quality)

    return categories


def merge_categories(predicted: dict[str, list[str]], added: dict[str, list[str]]) -> list[str]:
    ordered_keys = ["special", "character", "copyright", "artist", "general", "meta", "rating", "quality"]
    combined = []
    for key in ordered_keys:
        combined.extend(dedupe_preserve_order(added[key] + predicted[key]))
    return dedupe_preserve_order(combined)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone minimal WD14 tagger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="image directory to tag")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="HF mirror endpoint")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="WD14 model repo id")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="directory to store WD14 model files")
    parser.add_argument("--batch_size", type=int, default=4, help="inference batch size")
    parser.add_argument("--caption_extension", default=".txt", help="output caption file extension")
    parser.add_argument("--meta_tag_blacklist_file", default=str(DEFAULT_BLACKLIST), help="path to the meta-tag blacklist file")
    parser.add_argument("--force_download", action="store_true", help="force re-download the WD14 model")

    parser.add_argument("--general_threshold", type=float, default=0.35, help="threshold for general tags")
    parser.add_argument("--character_threshold", type=float, default=0.85, help="threshold for character tags")
    parser.add_argument("--copyright_threshold", type=float, default=0.35, help="threshold for copyright tags")
    parser.add_argument("--artist_threshold", type=float, default=0.35, help="threshold for artist tags")
    parser.add_argument("--meta_threshold", type=float, default=0.35, help="threshold for meta tags")
    parser.add_argument("--remove_underscore", action="store_true", default=True, help="replace underscores with spaces")
    parser.add_argument("--use_rating_tags", action="store_true", default=True, help="include rating tags")
    parser.add_argument("--use_quality_tags", action="store_true", default=True, help="include quality tags")

    parser.add_argument("--add_special_tags", default=None, help="comma-separated special tags")
    parser.add_argument("--add_character_tags", default=None, help="comma-separated character tags")
    parser.add_argument("--add_copyright_tags", default=None, help="comma-separated copyright tags")
    parser.add_argument("--add_artist_tags", default=None, help="comma-separated artist tags")
    parser.add_argument("--add_general_tags", default=None, help="comma-separated general tags")
    parser.add_argument("--add_meta_tags", default=None, help="comma-separated meta tags")
    parser.add_argument("--add_rating_tags", default=None, help="comma-separated rating tags")
    parser.add_argument("--add_quality_tags", default=None, help="comma-separated quality tags")
    return parser


def main() -> None:
    args = setup_parser().parse_args()

    os.environ["HF_ENDPOINT"] = args.mirror
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/huggingface/hub")

    input_dir = resolve_path(args.input_dir)
    model_dir = resolve_path(args.model_dir)
    blacklist_path = resolve_path(args.meta_tag_blacklist_file)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")

    model_location, onnx_model_path, default_format = download_model(args.repo_id, model_dir, args.force_download)
    metadata = load_tag_metadata(model_location, default_format, args.remove_underscore)
    session = make_session(onnx_model_path)
    input_name = session.get_inputs()[0].name

    images = list_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    undesired_tags = load_blacklist(blacklist_path)
    added_categories = {
        "special": [tag for tag in parse_tag_list(args.add_special_tags) if tag not in undesired_tags],
        "character": [tag for tag in parse_tag_list(args.add_character_tags) if tag not in undesired_tags],
        "copyright": [tag for tag in parse_tag_list(args.add_copyright_tags) if tag not in undesired_tags],
        "artist": [tag for tag in parse_tag_list(args.add_artist_tags) if tag not in undesired_tags],
        "general": [tag for tag in parse_tag_list(args.add_general_tags) if tag not in undesired_tags],
        "meta": [tag for tag in parse_tag_list(args.add_meta_tags) if tag not in undesired_tags],
        "rating": [tag for tag in parse_tag_list(args.add_rating_tags) if tag not in undesired_tags],
        "quality": [tag for tag in parse_tag_list(args.add_quality_tags) if tag not in undesired_tags],
    }
    thresholds = {
        "general": args.general_threshold,
        "character": args.character_threshold,
        "copyright": args.copyright_threshold,
        "artist": args.artist_threshold,
        "meta": args.meta_threshold,
    }

    for start in tqdm(range(0, len(images), args.batch_size), desc="Tagging"):
        batch_paths = images[start : start + args.batch_size]
        batch = np.stack([preprocess_image(Image.open(path)) for path in batch_paths]).astype(np.float32)
        if not default_format:
            batch = batch.transpose(0, 3, 1, 2)
            batch = batch / 127.5 - 1.0

        probs = session.run(None, {input_name: batch})[0]

        for image_path, prob in zip(batch_paths, probs):
            if default_format:
                predicted = categorize_default_format(
                    prob,
                    metadata,
                    thresholds,
                    include_quality=args.use_quality_tags,
                    include_rating=args.use_rating_tags,
                    undesired_tags=undesired_tags,
                )
            else:
                predicted = categorize_extended_format(
                    prob,
                    metadata,
                    thresholds,
                    include_quality=args.use_quality_tags,
                    include_rating=args.use_rating_tags,
                    undesired_tags=undesired_tags,
                )

            final_tags = merge_categories(predicted, added_categories)
            output_path = image_path.with_suffix(args.caption_extension)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(", ".join(final_tags) + "\n")

    print(f"Tagged {len(images)} images in {input_dir}")


if __name__ == "__main__":
    main()
