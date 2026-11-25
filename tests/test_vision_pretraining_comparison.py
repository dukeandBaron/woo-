"""Paired held-out reference reconstruction, not detector or test performance."""

from __future__ import annotations

import importlib
import importlib.util
import math

import numpy as np
import pytest

from tests.test_vision_pretraining_reference import (
    fit,
    fixture_recipe,
    load_reference,
    sha,
)
from visiondata_gate.audit_envelope import canonical_jcs_bytes
from visiondata_gate.vision_pretraining_numpy import (
    initial_weights,
    masked_examples,
    patchify,
    predict,
    read_checkpoint,
    sampled_mask,
)


def comparison_module():
    name = "visiondata_gate.vision_pretraining_comparison"
    assert importlib.util.find_spec(name) is not None, (
        "paired reconstruction validator is absent"
    )
    return importlib.import_module(name)


@pytest.fixture
def measured_reference(tmp_path):
    frozen, recipe = fixture_recipe(tmp_path)
    output = tmp_path / "run"
    return frozen, recipe, output, fit(load_reference(), frozen, recipe, output)


def test_validation_uses_one_image_one_mask_and_same_targets_in_both_arms(
    measured_reference, monkeypatch
):
    comparison = comparison_module()
    frozen, recipe, output, run = measured_reference
    reads = []
    real_read = comparison.read_pinned_patches

    def record_model_input(root, member, proposal):
        reads.append((member["sample_id"], member["split"]))
        return real_read(root, member, proposal)

    monkeypatch.setattr(comparison, "read_pinned_patches", record_model_input)
    before = {file.name: sha(file.read_bytes()) for file in output.iterdir()}
    result = comparison.compare_reference_on_validation(
        frozen,
        recipe,
        recipe["receipt_sha256"],
        output,
        run["receipt_sha256"],
        mask_seed=983,
    )
    assert reads == [("v0", "val")]
    assert result["denominator"] == {
        "validation_images": 1,
        "masked_patches": 10,
        "masked_rgb_scalar_targets": 7680,
    }
    assert result["metric"] == "MASKED_RGB_PIXEL_MSE"
    assert result["reference_run_receipt_sha256"] == run["receipt_sha256"]
    assert (
        result["baseline"]["checkpoint_sha256"] == run["checkpoint"]["initial_sha256"]
    )
    assert result["pretrained"]["checkpoint_sha256"] == run["checkpoint"]["sha256"]
    assert (
        result["test_evaluation"]
        == result["downstream_detection_evaluation"]
        == "NOT_RUN"
    )
    assert result["production_release_allowed"] is False
    patches = patchify((frozen / "images" / "val" / "v0.png").read_bytes(), 64, 16)
    inputs, targets = masked_examples(patches, sampled_mask(16, 10, 983, 0, 0))
    for channel, weights in (
        ("baseline", initial_weights(16, 19)),
        ("pretrained", read_checkpoint((output / "checkpoint.f32").read_bytes(), 16)),
    ):
        expected = np.mean((predict(inputs, weights) - targets) ** 2, dtype="float64")
        assert result[channel]["masked_mse"] == pytest.approx(expected, abs=1e-10)
    body = {key: val for key, val in result.items() if key != "receipt_sha256"}
    assert result["receipt_sha256"] == sha(canonical_jcs_bytes(body))
    assert {file.name: sha(file.read_bytes()) for file in output.iterdir()} == before
    assert math.isfinite(result["delta_pretrained_minus_random"])


def test_comparison_rejects_wrong_reference_receipt_and_tampered_weights(
    measured_reference,
):
    comparison = comparison_module()
    reference = load_reference()
    frozen, recipe, output, run = measured_reference
    with pytest.raises(
        reference.ReferencePretrainingError, match="RUN_RECEIPT_INVALID"
    ):
        comparison.compare_reference_on_validation(
            frozen, recipe, recipe["receipt_sha256"], output, "a" * 64, mask_seed=983
        )
    checkpoint = output / "checkpoint.f32"
    bounds = checkpoint.read_bytes()
    checkpoint.write_bytes(bounds[:-1] + bytes([bounds[-1] ^ 1]))
    with pytest.raises(reference.ReferencePretrainingError, match="CHECKPOINT_INVALID"):
        comparison.compare_reference_on_validation(
            frozen,
            recipe,
            recipe["receipt_sha256"],
            output,
            run["receipt_sha256"],
            mask_seed=983,
        )
