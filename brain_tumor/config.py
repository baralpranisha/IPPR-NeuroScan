from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "brain_tumor_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "brain_tumor_cnn.pt"
DEFAULT_IMAGE_SIZE = 224
CLASS_NAMES = ("healthy", "tumor")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}