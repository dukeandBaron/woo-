"""A real CPU fit on test-owned data, with off-product safety invariants."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pytest

from visiondata_gate.audit_envelope import canonical_jcs_bytes
from visiondata_gate.learning_detection_dataset import (
    freeze_detection_dataset,
    manifest_sha256,
)
from visiondata_gate.vision_pretraining_recipe import (
    VisionPretrainingRecipeInputs,
    build_pretraining_recipe,
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_reference():
    name = "visiondata_gate.vision_pretraining_reference"
    assert importlib.util.find_spec(name) is not None, "offline CPU executor is absent"
    return importlib.import_module(name)


def fixture_recipe(root: Path, *, holdout_shift: int = 0):
    source = root / "source"
    source.mkdir(parents=True)
    version = "synthetic-cpu-reference-v1"
    y, x = np.indices((24, 24))
    samples = []
    for index, (name, split) in enumerate(
        (
            ("t0", "train"),
            ("t1", "train"),
            ("t2", "train"),
            ("v0", "val"),
            ("s0", "test"),
        )
    ):
        base = 28 + index * 20 + (holdout_shift if split != "train" else 0)
        rgb = np.stack((base + x * 2, base + y * 2, base + (x + y) // 2), axis=-1)
        path = source / f"{name}.png"
        Image.fromarray(rgb.astype("uint8")).save(path)
        samples.append(
            {
                "sample_id": name,
                "image_path": path.name,
                "image_sha256": sha(path.read_bytes()),
                "split": split,
                "group_id": f"fixture-unit-{name}",
                "annotation_revision": 1,
                "reviewer_name": "Synthetic fixture",
                "reviewed": True,
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
        "source_version": version,
        "class_names": ["fixture"],
        "samples": samples,
    }
    target = root / "frozen"
    frozen = freeze_detection_dataset(
        source, manifest, target, manifest_sha256(manifest)
    )
    # The earlier API round separately verifies the live data pool. This fixture
    # tests only the offline run starting at a pinned reviewed recipe.
    binding = {
        "pool_id": "pool_synthetic",
        "receipt_sha256": "b" * 64,
        "version_id": version,
        "version_receipt_sha256": "c" * 64,
        "snapshot_id": "snapshot_synthetic",
        "annotation_bindings_sha256": "d" * 64,
    }
    proposal = VisionPretrainingRecipeInputs(
        dataset_id="dataset_synthetic",
        expected_dataset_receipt_sha256=frozen["receipt_sha256"],
        expected_pool_binding_sha256=binding["receipt_sha256"],
        budget={"epochs": 2, "batch": 2, "seed": 19, "threads": 1, "max_seconds": 20},
    )
    return target, build_pretraining_recipe(
        "synthetic_project", proposal, frozen, binding
    )


def fit(reference, frozen, recipe, output):
    return reference.run_reference_pretraining(
        frozen, recipe, recipe["receipt_sha256"], output, allow_reference_execution=True
    )


def test_reference_writes_real_weights_and_a_sealed_offline_receipt(tmp_path):
    reference = load_reference()
    frozen, recipe = fixture_recipe(tmp_path)
    output = tmp_path / "attempt"
    actual = fit(reference, frozen, recipe, output)
    assert actual["status"] == "REFERENCE_COMPLETED"
    assert actual["recipe_receipt_sha256"] == recipe["receipt_sha256"]
    assert actual["lineage"] == recipe["lineage"]
    assert actual["gradient_split"] == "train"
    assert actual["training_image_count"] == 3
    assert actual["sample_visits"] == 6
    assert actual["optimizer_steps"] == 4
    assert actual["runtime_executed"] is True
    assert actual["device"] == "cpu"
    assert actual["test_evaluation"] == actual["validation_evaluation"] == "NOT_RUN"
    assert actual["live_pool_authorization_checked"] is False
    assert actual["production_release_allowed"] is False
    assert actual["weight_transfer_compatibility"] == "NOT_VERIFIED"
    assert len(actual["training_mse_by_epoch"]) == 2
    assert all(
        0 <= loss < 1 and math.isfinite(loss)
        for loss in actual["training_mse_by_epoch"]
    )
    assert actual["checkpoint"]["sha256"] != actual["checkpoint"]["initial_sha256"]
    assert actual["checkpoint"]["round_trip"] == "VERIFIED"
    assert actual["checkpoint"]["format"] == "FLOAT32_LE_FIXED_LAYOUT_V1"
    checkpoint = output / "checkpoint.f32"
    assert checkpoint.stat().st_size == actual["checkpoint"]["byte_count"]
    assert sha(checkpoint.read_bytes()) == actual["checkpoint"]["sha256"]
    assert json.loads((output / "RUN_RECEIPT.json").read_bytes()) == actual
    assert (
        sha(
            canonical_jcs_bytes(
                {k: v for k, v in actual.items() if k != "receipt_sha256"}
            )
        )
        == actual["receipt_sha256"]
    )
    with pytest.raises(reference.ReferencePretrainingError, match="OUTPUT_EXISTS"):
        fit(reference, frozen, recipe, output)
    assert sha(checkpoint.read_bytes()) == actual["checkpoint"]["sha256"]


def test_reference_weight_bytes_do_not_change_when_only_holdouts_change(tmp_path):
    reference = load_reference()
    frozen_a, recipe_a = fixture_recipe(tmp_path / "a")
    frozen_b, recipe_b = fixture_recipe(tmp_path / "b", holdout_shift=13)
    assert (
        recipe_a["lineage"]["dataset_receipt_sha256"]
        != recipe_b["lineage"]["dataset_receipt_sha256"]
    )
    assert (
        recipe_a["data"]["train_image_manifest_sha256"]
        == recipe_b["data"]["train_image_manifest_sha256"]
    )
    a = fit(reference, frozen_a, recipe_a, tmp_path / "run-a")
    b = fit(reference, frozen_b, recipe_b, tmp_path / "run-b")
    assert a["checkpoint"]["sha256"] == b["checkpoint"]["sha256"]
    assert a["training_mse_by_epoch"] == b["training_mse_by_epoch"]


def test_reference_holds_without_consent_or_a_current_frozen_artifact(tmp_path):
    reference = load_reference()
    frozen, recipe = fixture_recipe(tmp_path)
    output = tmp_path / "attempt"
    with pytest.raises(
        reference.ReferencePretrainingError, match="EXPLICIT_OFFLINE_CONSENT_REQUIRED"
    ):
        reference.run_reference_pretraining(
            frozen,
            recipe,
            recipe["receipt_sha256"],
            output,
            allow_reference_execution=False,
        )
    with pytest.raises(reference.ReferencePretrainingError, match="RECIPE_RECEIPT"):
        reference.run_reference_pretraining(
            frozen, recipe, "f" * 64, output, allow_reference_execution=True
        )
    (frozen / "labels" / "val" / "v0.txt").write_text(
        "not the frozen label", encoding="utf-8"
    )
    with pytest.raises(
        reference.ReferencePretrainingError, match="FROZEN_DATASET_INVALID"
    ):
        fit(reference, frozen, recipe, output)
    assert not output.exists()


def test_masked_pixels_are_targets_and_cannot_leak_into_model_inputs():
    name = "visiondata_gate.vision_pretraining_numpy"
    assert importlib.util.find_spec(name) is not None, "reference mask kernel is absent"
    patch_model = importlib.import_module(name)
    original = np.ones((4, 2, 3), dtype="float32")
    hidden = np.array([1, 3])
    changed = original.copy()
    changed[hidden] = 0.25
    first_input, first_target = patch_model.masked_examples(original, hidden)
    second_input, second_target = patch_model.masked_examples(changed, hidden)
    assert np.array_equal(first_input, second_input)
    assert not np.array_equal(first_target, second_target)


def test_reference_worker_is_stopped_at_a_hard_deadline(tmp_path):
    reference = load_reference()
    child = tmp_path / "slow-worker.py"
    child.write_text("from time import sleep\nsleep(20)\n", encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(
        reference.ReferencePretrainingError, match="TIME_BUDGET_EXCEEDED"
    ):
        reference._bounded_worker(
            [sys.executable, "-B", str(child)], tmp_path, 1, started + 0.4
        )
    assert time.monotonic() - started < 6


def test_reference_receipt_cannot_pick_up_a_late_caller_mutation(tmp_path, monkeypatch):
    reference = load_reference()
    frozen, recipe = fixture_recipe(tmp_path)
    pinned = deepcopy(recipe)
    real_worker = reference._bounded_worker

    def mutate_after_fit(*args):
        status = real_worker(*args)
        recipe["lineage"]["version_id"] = "changed_after_fit"
        return status

    monkeypatch.setattr(reference, "_bounded_worker", mutate_after_fit)
    result = fit(reference, frozen, recipe, tmp_path / "attempt")
    assert result["recipe_receipt_sha256"] == pinned["receipt_sha256"]
    assert result["lineage"] == pinned["lineage"]
