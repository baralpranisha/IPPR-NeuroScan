from __future__ import annotations

import argparse
import json

from brain_tumor.training import TrainingConfig, train_model
from brain_tumor.visualization import generate_result_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Brain Tumor CNN on the Kaggle dataset."
    )
    parser.add_argument("--dataset-dir", required=True, help="Extracted dataset root.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_model(
        TrainingConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    )
    generate_result_plots(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()