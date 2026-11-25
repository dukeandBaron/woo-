"""Bounded offline reference fit; no product authorization or workbench route."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

if __name__ == "__main__" and not __package__:
    # The isolated child must import this checkout, not an older editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visiondata_gate.audit_envelope import canonical_jcs_bytes
from visiondata_gate.learning_detection_dataset import (
    DetectionDatasetError,
    _directory,
    _read,
    verify_detection_dataset,
)
from visiondata_gate.vision_pretraining_numpy import (
    BACKEND,
    LEARNING_RATE,
    optimize_train_patches,
    patchify,
    read_checkpoint,
)
from visiondata_gate.vision_pretraining_recipe import (
    VisionPretrainingRecipeInputs,
    build_pretraining_recipe,
)

_SHA256 = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,120}")
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_CHECKPOINT_BYTES = 128 * 1024
_CHECKPOINT = "checkpoint.f32"
RUN_RECEIPT_NAME = "RUN_RECEIPT.json"
_CODE = {
    "REFERENCE_RECIPE_RECEIPT_INVALID",
    "REFERENCE_FROZEN_DATASET_INVALID",
    "REFERENCE_TRAIN_MEMBER_CHANGED",
    "REFERENCE_OPTIMIZER_FAILED",
}


class ReferencePretrainingError(ValueError):
    """A sanitized failure before a reference result can be published."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sealed(value: dict) -> dict:
    return value | {"receipt_sha256": _sha(canonical_jcs_bytes(value))}


def snapshot_recipe(recipe: dict) -> dict:
    """Retain one immutable-by-ownership input for the worker and its receipt."""
    try:
        if not isinstance(recipe, dict):
            raise ValueError
        encoded = canonical_jcs_bytes(recipe)
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise ValueError
        return json.loads(encoded)
    except (ValueError, TypeError, OverflowError):
        raise ReferencePretrainingError("REFERENCE_RECIPE_RECEIPT_INVALID") from None


def _verified_recipe(recipe: dict, digest: str) -> VisionPretrainingRecipeInputs:
    try:
        if (
            not isinstance(recipe, dict)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError
        body = {key: value for key, value in recipe.items() if key != "receipt_sha256"}
        encoded = canonical_jcs_bytes(body)
        if (
            len(encoded) > _MAX_RECEIPT_BYTES
            or recipe.get("receipt_sha256") != digest
            or _sha(encoded) != digest
        ):
            raise ValueError
        lineage = recipe["lineage"]
        for key in ("pool_id", "version_id", "snapshot_id"):
            if not _ID.fullmatch(lineage[key]):
                raise ValueError
        for key in ("version_receipt_sha256", "annotation_bindings_sha256"):
            if not _SHA256.fullmatch(lineage[key]):
                raise ValueError
        if (
            recipe["schema_version"] != "visiondata-gate.vision-pretraining-recipe.v1"
            or recipe["status"] != "HOLD_NO_SELF_SUPERVISED_BACKEND"
            or not _ID.fullmatch(recipe["project_id"])
            or recipe["training_authorized"] is not False
        ):
            raise ValueError
        return VisionPretrainingRecipeInputs(
            dataset_id=lineage["dataset_id"],
            expected_dataset_receipt_sha256=lineage["dataset_receipt_sha256"],
            expected_pool_binding_sha256=lineage["pool_binding_sha256"],
            budget=recipe["budget"],
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        raise ReferencePretrainingError("REFERENCE_RECIPE_RECEIPT_INVALID") from None


def verify_recipe_snapshot(
    frozen_root: Path, recipe: dict, expected_receipt: str
) -> tuple[dict, VisionPretrainingRecipeInputs]:
    """Rebuild the recipe against checked files; does not recheck live pool authority."""
    proposal = _verified_recipe(recipe, expected_receipt)
    lineage = recipe["lineage"]
    try:
        frozen = verify_detection_dataset(
            frozen_root, proposal.expected_dataset_receipt_sha256
        )
    except (DetectionDatasetError, OSError, ValueError):
        raise ReferencePretrainingError("REFERENCE_FROZEN_DATASET_INVALID") from None
    if frozen["source_version"] != lineage["version_id"]:
        raise ReferencePretrainingError("REFERENCE_FROZEN_DATASET_INVALID")
    binding = {
        key: lineage[key]
        for key in (
            "pool_id",
            "version_id",
            "version_receipt_sha256",
            "snapshot_id",
            "annotation_bindings_sha256",
        )
    }
    binding["receipt_sha256"] = lineage["pool_binding_sha256"]
    rebuilt = build_pretraining_recipe(recipe["project_id"], proposal, frozen, binding)
    if canonical_jcs_bytes(rebuilt) != canonical_jcs_bytes(recipe):
        raise ReferencePretrainingError("REFERENCE_RECIPE_RECEIPT_INVALID")
    return frozen, proposal


def read_pinned_patches(
    root: Path, member: dict, proposal: VisionPretrainingRecipeInputs
):
    try:
        content = _read(root, member["image_path"])
        if _sha(content) != member["image_sha256"]:
            raise ValueError
        return patchify(content, proposal.budget.image_size, proposal.budget.patch_size)
    except (ValueError, OSError):
        raise ReferencePretrainingError("REFERENCE_TRAIN_MEMBER_CHANGED") from None


def _bounded_worker(
    command: list[str], folder: Path, threads: int, deadline: float
) -> int:
    """Wait for only our child; subprocess.run kills and reaps it on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ReferencePretrainingError("REFERENCE_TIME_BUDGET_EXCEEDED")
    env = {
        key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ
    }
    env.update(
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS=str(threads),
        CUDA_VISIBLE_DEVICES="",
    )
    try:
        return subprocess.run(
            command,
            cwd=folder,
            env=env,
            timeout=remaining,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).returncode
    except subprocess.TimeoutExpired:
        raise ReferencePretrainingError("REFERENCE_TIME_BUDGET_EXCEEDED") from None
    except OSError:
        raise ReferencePretrainingError("REFERENCE_WORKER_START_FAILED") from None


def _metrics(folder: Path, return_code: int, budget, training_count: int) -> dict:
    try:
        report = json.loads(_read(folder, "worker-result.json", _MAX_RECEIPT_BYTES))
        if return_code:
            code = report.get("error")
            if isinstance(code, str) and code in _CODE:
                raise ReferencePretrainingError(code)
            raise ValueError
        if set(report) != {
            "optimizer_steps",
            "numpy_version",
            "training_mse_by_epoch",
            "initial_sha256",
        }:
            raise ValueError
        losses = report["training_mse_by_epoch"]
        if (
            len(losses) != budget.epochs
            or any(
                type(v) not in (float, int) or not math.isfinite(v) or v < 0
                for v in losses
            )
            or type(report["optimizer_steps"]) is not int
            or report["optimizer_steps"]
            != math.ceil(training_count / budget.batch) * budget.epochs
            or not _SHA256.fullmatch(report["initial_sha256"])
            or not re.fullmatch(r"[0-9A-Za-z.+-]{1,64}", report["numpy_version"])
        ):
            raise ValueError
        return report
    except ReferencePretrainingError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        raise ReferencePretrainingError("REFERENCE_WORKER_RESULT_INVALID") from None


def run_reference_pretraining(
    frozen_root: Path,
    recipe: dict,
    expected_recipe_sha256: str,
    output_dir: Path,
    *,
    allow_reference_execution: bool,
) -> dict:
    """Fit one offline reference on authorized frozen files. Never queues a product run."""
    if allow_reference_execution is not True:
        raise ReferencePretrainingError("REFERENCE_EXPLICIT_OFFLINE_CONSENT_REQUIRED")
    started = time.monotonic()
    recipe = snapshot_recipe(recipe)
    proposal = _verified_recipe(recipe, expected_recipe_sha256)
    budget = proposal.budget
    deadline = started + budget.max_seconds
    try:
        source = _directory(frozen_root)
        candidate = Path(os.path.abspath(output_dir))
        output = _directory(candidate.parent) / candidate.name
    except (ValueError, OSError):
        raise ReferencePretrainingError("REFERENCE_ROOT_UNAVAILABLE") from None
    if os.path.lexists(output):
        raise ReferencePretrainingError("REFERENCE_OUTPUT_EXISTS")
    if output.is_relative_to(source):
        raise ReferencePretrainingError("REFERENCE_OUTPUT_INSIDE_DATASET")
    with TemporaryDirectory(prefix=".vision-ref-", dir=output.parent) as temp:
        folder = Path(temp)
        request = {
            "frozen_root": str(source),
            "recipe": recipe,
            "expected_recipe_sha256": expected_recipe_sha256,
        }
        (folder / "worker-input.json").write_bytes(canonical_jcs_bytes(request))
        command = [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--reference-worker",
        ]
        status = _bounded_worker(command, folder, budget.threads, deadline)
        training_count = recipe["data"]["split_counts"]["train"]
        report = _metrics(folder, status, budget, training_count)
        try:
            data = _read(folder, _CHECKPOINT, _MAX_CHECKPOINT_BYTES)
            read_checkpoint(data, budget.patch_size)
        except (ValueError, OSError):
            raise ReferencePretrainingError("REFERENCE_CHECKPOINT_INVALID") from None
        elapsed_ms = round((time.monotonic() - started) * 1000)
        if time.monotonic() >= deadline:
            raise ReferencePretrainingError("REFERENCE_TIME_BUDGET_EXCEEDED")
        receipt = _sealed(
            {
                "schema_version": "visiondata-gate.vision-pretraining-reference-run.v1",
                "status": "REFERENCE_COMPLETED",
                "scope": "OFFLINE_REFERENCE_ONLY",
                "project_id": recipe["project_id"],
                "recipe_receipt_sha256": expected_recipe_sha256,
                "inputs_sha256": recipe["inputs_sha256"],
                "lineage": recipe["lineage"],
                "device": "cpu",
                "backend": BACKEND,
                "numpy_version": report["numpy_version"],
                "objective": recipe["objective"],
                "budget": budget.model_dump(mode="json"),
                "optimizer": {"name": "MINIBATCH_SGD", "learning_rate": LEARNING_RATE},
                "elapsed_ms": elapsed_ms,
                "gradient_split": "train",
                "train_image_manifest_sha256": recipe["data"][
                    "train_image_manifest_sha256"
                ],
                "training_image_count": training_count,
                "sample_visits": training_count * budget.epochs,
                "optimizer_steps": report["optimizer_steps"],
                "training_mse_by_epoch": report["training_mse_by_epoch"],
                "checkpoint": {
                    "relative_path": _CHECKPOINT,
                    "format": "FLOAT32_LE_FIXED_LAYOUT_V1",
                    "sha256": _sha(data),
                    "initial_sha256": report["initial_sha256"],
                    "byte_count": len(data),
                    "round_trip": "VERIFIED",
                },
                "runtime_executed": True,
                "live_pool_authorization_checked": False,
                "validation_evaluation": "NOT_RUN",
                "test_evaluation": "NOT_RUN",
                "weight_transfer_compatibility": "NOT_VERIFIED",
                "production_release_allowed": False,
                "machine_write_permitted": False,
            }
        )
        try:
            output.mkdir()
        except FileExistsError:
            raise ReferencePretrainingError("REFERENCE_OUTPUT_EXISTS") from None
        # The receipt is the completion marker. Failed publishes never report success.
        with (output / _CHECKPOINT).open("xb") as stream:
            stream.write(data)
        with (output / RUN_RECEIPT_NAME).open("xb") as stream:
            stream.write(canonical_jcs_bytes(receipt))
    return receipt


def _run_worker(folder: Path) -> int:
    import numpy as np

    try:
        request = json.loads(_read(folder, "worker-input.json", _MAX_RECEIPT_BYTES * 2))
        recipe = request["recipe"]
        root = _directory(Path(request["frozen_root"]))
        frozen, proposal = verify_recipe_snapshot(
            root, recipe, request["expected_recipe_sha256"]
        )
        members = sorted(
            (item for item in frozen["samples"] if item["split"] == "train"),
            key=lambda item: item["sample_id"],
        )
        patches = [read_pinned_patches(root, row, proposal) for row in members]
        initial, trained, losses, steps = optimize_train_patches(
            patches, proposal.budget, recipe["masked_patches_per_image"]
        )
        (folder / _CHECKPOINT).write_bytes(trained)
        result = {
            "optimizer_steps": steps,
            "numpy_version": np.__version__,
            "training_mse_by_epoch": losses,
            "initial_sha256": _sha(initial),
        }
        (folder / "worker-result.json").write_bytes(canonical_jcs_bytes(result))
        return 0
    except ReferencePretrainingError as failure:
        code = str(failure)
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError):
        code = "REFERENCE_OPTIMIZER_FAILED"
    (folder / "worker-result.json").write_bytes(canonical_jcs_bytes({"error": code}))
    return 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--reference-worker"]:
        raise SystemExit("Use the explicit offline reference API.")
    raise SystemExit(_run_worker(Path.cwd()))
