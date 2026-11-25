"""Matched validation reconstruction against the exact original initialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .audit_envelope import canonical_jcs_bytes
from .learning_detection_dataset import _directory, _read
from .vision_pretraining_numpy import (
    BACKEND,
    checkpoint_bytes,
    initial_weights,
    masked_examples,
    predict,
    read_checkpoint,
    sampled_mask,
)
from .vision_pretraining_reference import (
    RUN_RECEIPT_NAME,
    ReferencePretrainingError,
    _CHECKPOINT,
    _MAX_CHECKPOINT_BYTES,
    _MAX_RECEIPT_BYTES,
    _SHA256,
    _sealed,
    _sha,
    read_pinned_patches,
    snapshot_recipe,
    verify_recipe_snapshot,
)


def _verified_run(
    run_folder: Path, expected: str, recipe: dict, recipe_sha: str
) -> tuple[Path, dict]:
    try:
        root = _directory(run_folder)
        run = json.loads(_read(root, RUN_RECEIPT_NAME, _MAX_RECEIPT_BYTES))
        unsigned = {k: v for k, v in run.items() if k != "receipt_sha256"}
        if (
            not _SHA256.fullmatch(expected)
            or run.get("receipt_sha256") != expected
            or _sha(canonical_jcs_bytes(unsigned)) != expected
            or run["schema_version"]
            != "visiondata-gate.vision-pretraining-reference-run.v1"
            or run["status"] != "REFERENCE_COMPLETED"
            or run["scope"] != "OFFLINE_REFERENCE_ONLY"
            or run["device"] != "cpu"
            or run["backend"] != BACKEND
            or run["recipe_receipt_sha256"] != recipe_sha
            or run["project_id"] != recipe["project_id"]
            or run["inputs_sha256"] != recipe["inputs_sha256"]
            or run["lineage"] != recipe["lineage"]
            or run["budget"] != recipe["budget"]
            or run["gradient_split"] != "train"
            or run["train_image_manifest_sha256"]
            != recipe["data"]["train_image_manifest_sha256"]
            or run["training_image_count"] != recipe["data"]["split_counts"]["train"]
            or run["validation_evaluation"] != "NOT_RUN"
            or run["test_evaluation"] != "NOT_RUN"
            or run["weight_transfer_compatibility"] != "NOT_VERIFIED"
            or run["runtime_executed"] is not True
            or run["live_pool_authorization_checked"] is not False
            or run["production_release_allowed"] is not False
            or run["machine_write_permitted"] is not False
        ):
            raise ValueError
        return root, run
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        raise ReferencePretrainingError("REFERENCE_RUN_RECEIPT_INVALID") from None


def _verified_weights(root: Path, run: dict, patch_size: int, seed: int) -> tuple:
    try:
        checkpoint = run["checkpoint"]
        content = _read(root, _CHECKPOINT, _MAX_CHECKPOINT_BYTES)
        if (
            checkpoint["relative_path"] != _CHECKPOINT
            or checkpoint["format"] != "FLOAT32_LE_FIXED_LAYOUT_V1"
            or checkpoint["round_trip"] != "VERIFIED"
            or checkpoint["byte_count"] != len(content)
            or checkpoint["sha256"] != _sha(content)
        ):
            raise ValueError
        trained = read_checkpoint(content, patch_size)
    except (OSError, ValueError, TypeError, KeyError):
        raise ReferencePretrainingError("REFERENCE_CHECKPOINT_INVALID") from None
    baseline = initial_weights(patch_size, seed)
    if np.__version__ != run["numpy_version"] or _sha(
        checkpoint_bytes(baseline)
    ) != checkpoint.get("initial_sha256"):
        raise ReferencePretrainingError("REFERENCE_INITIALIZATION_NOT_REPRODUCIBLE")
    return baseline, trained


def compare_reference_on_validation(
    frozen_root: Path,
    recipe: dict,
    expected_recipe_sha256: str,
    run_dir: Path,
    expected_run_sha256: str,
    *,
    mask_seed: int,
) -> dict:
    """Evaluate both arms on identical val inputs; never fit or evaluate test."""
    if type(mask_seed) is not int or not 0 <= mask_seed <= 2_147_483_647:
        raise ReferencePretrainingError("REFERENCE_VALIDATION_MASK_SEED_INVALID")
    recipe = snapshot_recipe(recipe)
    results_folder, run = _verified_run(
        run_dir, expected_run_sha256, recipe, expected_recipe_sha256
    )
    frozen, proposal = verify_recipe_snapshot(
        frozen_root, recipe, expected_recipe_sha256
    )
    budget = proposal.budget
    baseline, trained = _verified_weights(
        results_folder, run, budget.patch_size, budget.seed
    )
    root = _directory(frozen_root)
    members = sorted(
        (row for row in frozen["samples"] if row["split"] == "val"),
        key=lambda row: row["sample_id"],
    )
    compared = []
    pixel_targets = 0
    errors = [0.0, 0.0]
    for index, member in enumerate(members):
        try:
            patches = read_pinned_patches(root, member, proposal)
        except ReferencePretrainingError:
            raise ReferencePretrainingError(
                "REFERENCE_VALIDATION_MEMBER_CHANGED"
            ) from None
        mask = sampled_mask(
            len(patches), recipe["masked_patches_per_image"], mask_seed, 0, index
        )
        features, target = masked_examples(patches, mask)
        for arm, weights in enumerate((baseline, trained)):
            delta = predict(features, weights) - target
            errors[arm] += float(np.sum(np.square(delta), dtype="float64"))
        pixel_targets += target.size
        compared.append(
            {
                "sample_id": member["sample_id"],
                "image_sha256": member["image_sha256"],
                "pixel_sha256": member["pixel_sha256"],
                "masked_patch_indices": mask.tolist(),
            }
        )
    if not pixel_targets or not all(math.isfinite(error) for error in errors):
        raise ReferencePretrainingError("REFERENCE_VALIDATION_METRIC_UNAVAILABLE")
    random_mse, trained_mse = (error / pixel_targets for error in errors)
    return _sealed(
        {
            "schema_version": "visiondata-gate.vision-pretraining-validation-comparison.v1",
            "status": "REFERENCE_VALIDATION_MEASURED",
            "scope": "OFFLINE_REFERENCE_ONLY",
            "recipe_receipt_sha256": expected_recipe_sha256,
            "reference_run_receipt_sha256": expected_run_sha256,
            "dataset_receipt_sha256": proposal.expected_dataset_receipt_sha256,
            "backend": BACKEND,
            "metric": "MASKED_RGB_PIXEL_MSE",
            "validation_mask_seed": mask_seed,
            "validation_mask_manifest_sha256": _sha(canonical_jcs_bytes(compared)),
            "denominator": {
                "validation_images": len(members),
                "masked_patches": len(members) * recipe["masked_patches_per_image"],
                "masked_rgb_scalar_targets": pixel_targets,
            },
            "baseline": {
                "initialization": "EXACT_IN_RUN_RANDOM_INITIALIZATION",
                "checkpoint_sha256": run["checkpoint"]["initial_sha256"],
                "masked_mse": random_mse,
            },
            "pretrained": {
                "initialization": "FITTED_ON_BOUND_TRAIN_SPLIT",
                "checkpoint_sha256": run["checkpoint"]["sha256"],
                "masked_mse": trained_mse,
            },
            "delta_pretrained_minus_random": trained_mse - random_mse,
            "test_evaluation": "NOT_RUN",
            "downstream_detection_evaluation": "NOT_RUN",
            "production_release_allowed": False,
            "machine_write_permitted": False,
        }
    )
