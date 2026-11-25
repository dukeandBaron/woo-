"""Read-only contract for a planned masked-patch reconstruction experiment."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .audit_envelope import canonical_jcs_bytes


class VisionPretrainingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    epochs: int = Field(default=1, ge=1, le=5)
    image_size: int = Field(default=64, ge=64, le=320, multiple_of=32)
    patch_size: Literal[8, 16, 32] = 16
    mask_basis_points: int = Field(default=6250, ge=2500, le=7500)
    batch: int = Field(default=2, ge=1, le=8)
    seed: int = Field(default=0, ge=0, le=2147483647)
    max_seconds: int = Field(default=120, ge=10, le=600)
    threads: int = Field(default=2, ge=1, le=4)


class VisionPretrainingRecipeInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,120}$")
    expected_dataset_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_pool_binding_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    budget: VisionPretrainingBudget = Field(default_factory=VisionPretrainingBudget)


def _sha(document) -> str:
    return hashlib.sha256(canonical_jcs_bytes(document)).hexdigest()


def build_pretraining_recipe(
    project: str, request: VisionPretrainingRecipeInputs, frozen: dict, binding: dict
) -> dict:
    """Describe what a future runner can use, without including paths or labels."""
    training_images = [
        {key: row[key] for key in ("sample_id", "image_sha256", "pixel_sha256")}
        for row in sorted(frozen["samples"], key=lambda item: item["sample_id"])
        if row["split"] == "train"
    ]
    budget = request.budget
    patches = (budget.image_size // budget.patch_size) ** 2
    value = {
        "schema_version": "visiondata-gate.vision-pretraining-recipe.v1",
        "project_id": project,
        "status": "HOLD_NO_SELF_SUPERVISED_BACKEND",
        "inputs_sha256": _sha(request.model_dump(mode="json")),
        "objective": "MASKED_PATCH_RECONSTRUCTION_V1",
        "device": "cpu",
        "budget": budget.model_dump(mode="json"),
        "patches_per_image": patches,
        "masked_patches_per_image": patches * budget.mask_basis_points // 10_000,
        "training_contract": {
            "image_transform": "RGB_BILINEAR_SQUARE_UNIT_INTERVAL",
            "mask_selector": "SEEDED_UNIFORM_WITHOUT_REPLACEMENT",
            "loss": "MASKED_PIXEL_MSE",
            "initialization": "RANDOM_NO_EXTERNAL_WEIGHTS",
        },
        "lineage": {
            "dataset_id": request.dataset_id,
            "dataset_receipt_sha256": frozen["receipt_sha256"],
            "pool_id": binding["pool_id"],
            "pool_binding_sha256": binding["receipt_sha256"],
            "version_id": binding["version_id"],
            "version_receipt_sha256": binding["version_receipt_sha256"],
            "snapshot_id": binding["snapshot_id"],
            "annotation_bindings_sha256": binding["annotation_bindings_sha256"],
        },
        "data": {
            "gradient_split": "train",
            "train_image_manifest_sha256": _sha(training_images),
            "split_counts": dict(frozen["split_counts"]),
            "labels_role": "PROVENANCE_ONLY",
            "validation_role": "RESERVED_NOT_USED",
            "test_role": "RESERVED_NOT_USED",
        },
        "downstream": {
            "task": "SUPERVISED_DETECTION",
            "checkpoint_status": "NOT_CREATED",
            "weight_transfer_compatibility": "NOT_VERIFIED",
            "comparison_baseline": "MATCHED_RANDOM_INITIALIZATION",
        },
        "training_authorized": False,
        "run_created": False,
        "runtime_executed": False,
        "test_evaluation": "NOT_RUN",
        "production_release_allowed": False,
        "machine_write_permitted": False,
    }
    return value | {"receipt_sha256": _sha(value)}
