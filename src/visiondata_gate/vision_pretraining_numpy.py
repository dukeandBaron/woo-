"""Small, fixed NumPy reference for masked-pixel reconstruction on a CPU."""

from __future__ import annotations

from io import BytesIO
from math import isqrt

import numpy as np
from PIL import Image

from .vision_pretraining_recipe import VisionPretrainingBudget

BACKEND = "NUMPY_CONTEXT_PATCH_MLP_V1"
LEARNING_RATE = 0.15
_FEATURES = 5  # RGB context from visible patches only, plus masked row/column.
_HIDDEN = 8


def _shapes(patch_size: int) -> tuple[tuple[int, ...], ...]:
    outputs = 3 * patch_size**2
    return (_FEATURES, _HIDDEN), (_HIDDEN,), (_HIDDEN, outputs), (outputs,)


def checkpoint_bytes(weights: tuple[np.ndarray, ...]) -> bytes:
    return b"".join(
        np.ascontiguousarray(value, dtype="<f4").tobytes() for value in weights
    )


def read_checkpoint(data: bytes, patch_size: int) -> tuple[np.ndarray, ...]:
    """Only fixed-size raw numeric arrays; no object, archive, or code loader."""
    shapes = _shapes(patch_size)
    sizes = tuple(int(np.prod(shape)) for shape in shapes)
    if len(data) != 4 * sum(sizes):
        raise ValueError("CHECKPOINT_SIZE_MISMATCH")
    flat = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(flat).all():
        raise ValueError("CHECKPOINT_NONFINITE")
    offset = 0
    weights = []
    for count, shape in zip(sizes, shapes, strict=True):
        weights.append(flat[offset : offset + count].reshape(shape))
        offset += count
    return tuple(weights)


def initial_weights(patch_size: int, seed: int) -> tuple[np.ndarray, ...]:
    w1, b1, w2, b2 = _shapes(patch_size)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, 0.15, w1).astype("float32"),
        np.zeros(b1, dtype="float32"),
        rng.normal(0, 0.04, w2).astype("float32"),
        np.full(b2, 0.5, dtype="float32"),
    )


def patchify(data: bytes, image_size: int, patch_size: int) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        resized = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype="float32") / np.float32(255)
    grid = image_size // patch_size
    return (
        pixels.reshape(grid, patch_size, grid, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid**2, patch_size**2, 3)
    )


def masked_examples(
    patches: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grid = isqrt(len(patches))
    visible = np.ones(len(patches), dtype=bool)
    visible[mask] = False
    context = patches[visible].mean(axis=(0, 1))
    features = np.empty((len(mask), _FEATURES), dtype="float32")
    features[:, :3] = context
    features[:, 3] = 2 * (mask // grid) / (grid - 1) - 1
    features[:, 4] = 2 * (mask % grid) / (grid - 1) - 1
    return features, patches[mask].reshape(len(mask), -1)


def sampled_mask(
    size: int, count: int, seed: int, pass_index: int, item_index: int
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, pass_index, item_index]))
    return np.sort(rng.choice(size, size=count, replace=False))


def predict(features: np.ndarray, weights: tuple[np.ndarray, ...]) -> np.ndarray:
    w1, b1, w2, b2 = weights
    return np.tanh(features @ w1 + b1) @ w2 + b2


def _step(
    weights: tuple[np.ndarray, ...], features: np.ndarray, target: np.ndarray
) -> tuple[float, int]:
    w1, b1, w2, b2 = weights
    hidden = np.tanh(features @ w1 + b1)
    errors = hidden @ w2 + b2 - target
    total = float(np.sum(np.square(errors), dtype="float64"))
    if not np.isfinite(total):
        raise ValueError("NONFINITE_TRAINING_LOSS")
    grad_output = errors * np.float32(2 / errors.size)
    grad_hidden = (grad_output @ w2.T) * (1 - hidden**2)
    rate = np.float32(LEARNING_RATE)
    w1 -= rate * (features.T @ grad_hidden)
    b1 -= rate * grad_hidden.sum(axis=0)
    w2 -= rate * (hidden.T @ grad_output)
    b2 -= rate * grad_output.sum(axis=0)
    return total, errors.size


def optimize_train_patches(
    train: list[np.ndarray], budget: VisionPretrainingBudget, masked_count: int
) -> tuple[bytes, bytes, list[float], int]:
    """Take already verified train-only tensors; no I/O or validation scoring."""
    weights = initial_weights(budget.patch_size, budget.seed)
    initial = checkpoint_bytes(weights)
    losses = []
    steps = 0
    for epoch in range(budget.epochs):
        total = count = 0
        for offset in range(0, len(train), budget.batch):
            examples = []
            for i in range(offset, min(offset + budget.batch, len(train))):
                patches = train[i]
                mask = sampled_mask(len(patches), masked_count, budget.seed, epoch, i)
                examples.append(masked_examples(patches, mask))
            features, target = (
                np.concatenate(parts) for parts in zip(*examples, strict=True)
            )
            error, denominator = _step(weights, features, target)
            total += error
            count += denominator
            steps += 1
        losses.append(total / count)
    return initial, checkpoint_bytes(weights), losses, steps
