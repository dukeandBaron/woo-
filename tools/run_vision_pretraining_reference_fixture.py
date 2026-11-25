"""Reproduce the offline CPU fit and validation pairing using generated images only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visiondata_gate.audit_envelope import canonical_jcs_bytes
from visiondata_gate.learning_detection_dataset import (
    _directory,
    freeze_detection_dataset,
    manifest_sha256,
)
from visiondata_gate.vision_pretraining_comparison import (
    compare_reference_on_validation,
)
from visiondata_gate.vision_pretraining_recipe import (
    VisionPretrainingRecipeInputs,
    build_pretraining_recipe,
)
from visiondata_gate.vision_pretraining_reference import run_reference_pretraining


def synthetic_recipe(output: Path) -> tuple[Path, dict]:
    source = output / "generated"
    source.mkdir()
    version = "synthetic-reference-r006-v1"
    rows, cols = np.indices((32, 32))
    samples = []
    for index, split in enumerate(["train"] * 8 + ["val"] * 2 + ["test"] * 2):
        name = f"generated-{index:02d}"
        # Fixed family; uniqueness comes from a different generated lighting
        # offset for every image, not from re-encoding the same RGB pixels.
        image = np.stack(
            (
                48 + index * 2 + 3 * cols,
                68 + index * 2 + 2 * rows,
                86 + index * 2 + rows + cols,
            ),
            axis=2,
        ).astype("uint8")
        path = source / f"{name}.png"
        Image.fromarray(image).save(path)
        samples.append(
            {
                "sample_id": name,
                "image_path": path.name,
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "group_id": f"generated-group-{index:02d}",
                "split": split,
                "reviewed": True,
                "reviewer_name": "Local synthetic fixture",
                "annotation_revision": 1,
                "boxes": [
                    {
                        "class_id": 0,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ],
            }
        )
    manifest = {
        "schema_version": "visiondata-gate.detection-dataset.v1",
        "class_names": ["synthetic_fixture"],
        "source_version": version,
        "samples": samples,
    }
    frozen = output / "frozen"
    dataset = freeze_detection_dataset(
        source, manifest, frozen, manifest_sha256(manifest)
    )
    # Explicit fixture lineage; this does not assert that a live pool has reviewed it.
    binding = {
        "pool_id": "synthetic_reference_pool",
        "version_id": version,
        "snapshot_id": "synthetic_reference_snapshot",
        "receipt_sha256": "b" * 64,
        "version_receipt_sha256": "c" * 64,
        "annotation_bindings_sha256": "d" * 64,
    }
    proposal = VisionPretrainingRecipeInputs(
        dataset_id="synthetic_reference_dataset",
        expected_dataset_receipt_sha256=dataset["receipt_sha256"],
        expected_pool_binding_sha256=binding["receipt_sha256"],
        budget={"epochs": 5, "batch": 2, "seed": 19, "threads": 1, "max_seconds": 30},
    )
    return frozen, build_pretraining_recipe(
        "synthetic_reference_project", proposal, dataset, binding
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory under an existing local parent",
    )
    parser.add_argument(
        "--allow-reference-execution",
        action="store_true",
        help="explicitly opt in to the synthetic CPU fit",
    )
    options = parser.parse_args()
    if not options.allow_reference_execution:
        parser.error("--allow-reference-execution is required")
    absolute = options.output_dir.absolute()
    output = _directory(absolute.parent) / absolute.name
    output.mkdir(exist_ok=False)
    frozen, recipe = synthetic_recipe(output)
    (output / "RECIPE.json").write_bytes(canonical_jcs_bytes(recipe))
    fit_output = output / "fit"
    run = run_reference_pretraining(
        frozen,
        recipe,
        recipe["receipt_sha256"],
        fit_output,
        allow_reference_execution=True,
    )
    measured = compare_reference_on_validation(
        frozen,
        recipe,
        recipe["receipt_sha256"],
        fit_output,
        run["receipt_sha256"],
        mask_seed=983,
    )
    (output / "VALIDATION_COMPARISON.json").write_bytes(canonical_jcs_bytes(measured))
    summary = {
        "scope": "SYNTHETIC_REFERENCE_ONLY",
        "split_counts": recipe["data"]["split_counts"],
        "fit_epochs": run["budget"]["epochs"],
        "optimizer_steps": run["optimizer_steps"],
        "fit_elapsed_ms": run["elapsed_ms"],
        "checkpoint_sha256": run["checkpoint"]["sha256"],
        "baseline_mse": measured["baseline"]["masked_mse"],
        "pretrained_mse": measured["pretrained"]["masked_mse"],
        "validation_denominator": measured["denominator"],
        "validation_mask_manifest_sha256": measured["validation_mask_manifest_sha256"],
        "comparison_receipt_sha256": measured["receipt_sha256"],
        "test_evaluation": "NOT_RUN",
        "downstream_detection_evaluation": "NOT_RUN",
    }
    (output / "SUMMARY.json").write_bytes(canonical_jcs_bytes(summary))
    print(json.dumps(summary, skipkeys=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
