# Industrial vision training platform — implementation plan

**Goal:** Connect the existing annotation and governed-dataset workflow to inspectable training recipes, controlled local execution, independent evaluation, and reviewed feedback.

**Architecture:** Extend the existing Data Pool, detection dataset and local runtime contracts. Separate planning from authorization and execution. Use the current split/label/weight identities; do not create a second dataset registry.

**Stack:** FastAPI local routes, Pydantic contracts, SQLite receipts, existing bounded CPU YOLO worker, existing React workbench.

## Round policy

Target: **100 evidence-backed engineering iterations**. Count one round only when it has a defined gap, a frozen baseline, a meaningful change or tested negative result, and a reproducible check or measured comparison. A commit, green mock, or dry run is not a training result. Do not auto-merge, start remote/GPU jobs, broaden data sharing, or promote models from this plan.

## R001: Validate a training proposal without launching it

- Baseline: POST /vision-training-runs queues a real task. The governed-Data-Pool integration test formerly used that call as “preflight”; there was no independent read-only route.
- First add regression tests in tests/test_vision_data_pool_bridge.py and tests/test_vision_model_api.py. Verify that a valid governed proposal returns a stable sealed receipt with no run/table initialization allowed, that invalid input/unauthorized requests still fail, and that the existing launch remains a separate approval. A nonexistent route should be observed as the failing baseline.
- Implement the read-only proposal DTO and verifier in src/visiondata_gate/local_model_registry.py. Reuse the existing frozen dataset, runtime and weight validations instead of duplicating them. Expose the localhost/member-scoped route in src/visiondata_gate/vision_model_api.py.
- Verify: pytest tests/test_vision_data_pool_bridge.py tests/test_vision_model_api.py tests/test_vision_training_lifecycle.py -q. Add the detection dataset and feedback test modules to check surrounding behavior. Report skips and test-double scope separately.

### R001 verification record — completed (1 of 100)

- Baseline: public main at `b9245259b023b61b99c1384e7708c9cf9b75f2fb`. Three red tests observed: the proposal route returned 404 and the proposal contract was absent.
- Green: five focused cases passed, including a real governed-pool/frozen-label fixture, a cold registry, local access and authentication, and a changed registered weight file. Preview created no training job or authorization record.
- Surrounding regression: `test_vision_training_lifecycle.py`, `test_learning_detection_dataset.py` and `test_vision_feedback_lifecycle.py` reported 121 passed, 1 skipped (Windows host does not permit creating a symlink). This is not a full-repository test. The warm-start preflight case is present in both runs.
- The published training-launch request's fields, default values, constraints and required fields were verified unchanged. No live trainer, external runtime, GPU, self-supervised pretraining, independent test-set evaluation or production deployment was run in this round.

## R002: Require a fresh workbench preflight before human-authorized submission

- Baseline: `144601dd` supported read-only backend preflight, but the workbench could submit without it. Three contract tests failed for the missing client operation and the Edge test failed for the missing preview button.
- Validate the exact proposal and strong response seal in the TypeScript client. Compare the project, proposal digest, budget and registered data/runtime/weight identities. Reject contradictory execution/release/authorization claims.
- Keep the receipt in the current draft only. A change of inputs, a resource refresh, or a new login generation discards the previous receipt; returning to the old draft does not revive old requests. No write lock or saved browser receipt is introduced by preview.
- Green: 20/20 Node vision contract tests and 26/26 full Edge visual-model workbench tests (synthetic transport, real forms); TypeScript checks and Vite production build passed. Tests exercise data/weights/budget invalidation, delayed responses, failure, and the unchanged separate human authorization.
- Scope: the workbench is gated; the existing backend launch API still enforces its own checks and human grants. No live training was launched.

## R003: Bind a read-only self-supervised recipe to current reviewed data

- Use a distinct local, project-member recipe route and a closed parameter set for masked-patch reconstruction. It must not accept an executable, model file, training grant or free-form configuration.
- Verify the registered dataset bytes against its receipt and recheck the current pool version, authorization and annotation lineage. Bind both the annotation receipt lineage and the pixel identity of the train-only manifest. Keep validation/test reserved and labels provenance-only.
- Freeze a declared CPU budget, patch size, deterministic seed and mask count. Preserve the future comparison baseline (matched random initialization); a downstream transfer adapter and checkpoint do not exist yet.
- Red: 7 targeted checks failed against the absent route. Green: the same 7 checks passed (100 deselected) using isolated fixtures, including the real Gate/Data Pool producer over synthetic images. The positive fixture retained 2 train / 1 val / 1 test and produced a stable sealed recipe. Negative cases covered non-local access, missing identity or registry, legacy unbound input, mismatched digests, source revocation and modified frozen holdout labels.
- The sealed recipe reports `HOLD_NO_SELF_SUPERVISED_BACKEND`, not an executable preflight, checkpoint, evaluation or release. The positive test found no new vision ledger entry, run, or file.

## R004: Show the verified held recipe in the model workbench

- Red: the contract test observed no client operation and the Edge test observed no workbench recipe entry.
- The supervised form and the separate recipe panel share the same frozen dataset selection but have independent budgets and receipts. The client independently verifies the annotation snapshot lineage, exact recipe parameters, split counts and train-image digest. Selecting an external manifest leaves recipe generation disabled.
- A changed dataset, recipe budget, project or login generation drops the receipt. The read-only recipe cannot satisfy supervised preflight, cause a launch, or acquire a write lock.
- Green: 21/21 Node vision contract tests, 27/27 full Edge workbench tests (synthetic transport), and the TypeScript/Vite production build passed. The recipe stays HOLD; no self-supervised job or checkpoint exists.

## R005: Execute the recipe as a bounded offline CPU reference

- Baseline `f651c9d`: a governed recipe can be reviewed, but no optimizer or reference weights exist. The public model workbench stays HOLD throughout this round.
- Test first in `tests/test_vision_pretraining_reference.py`: real synthetic frozen data and one finite optimizer run, fixed-seed holdout substitution, missing consent and stale/tampered inputs, and a killed/reaped worker after deadline.
- Implement a small NumPy contextual patch model and fixed-size numeric checkpoint in `visiondata_gate.vision_pretraining_numpy`. Reuse the held recipe's RGB resize, explicit masks, and masked-pixel MSE; keep labels and non-train images outside the optimizer.
- Implement `visiondata_gate.vision_pretraining_reference`: explicit offline consent, pinned recipe/frozen identities, process deadline/CPU thread cap, new output directory, sealed checkpoint receipt. Reopen only numeric weights from the expected byte layout. Never grant live project, model selection or production authority.
- Verify with `pytest tests/test_vision_pretraining_reference.py tests/test_learning_detection_dataset.py -q`, and a deterministic synthetic smoke with fixed data, budget and seed. Do not count a dry run or a training loss as held-out improvement.
- Red: five missing-engine checks failed; a separate RED showed a caller mutation after training could enter the receipt until the draft was copied. Final green: 66 passed / 1 Windows symlink skip across the reference and frozen-dataset suites.
- Measured synthetic reference: 3 train / 1 val / 1 test; 2 fit epochs, 6 visits, 4 optimizer steps; 27,840 checkpoint bytes. Replacing both holdouts kept the checkpoint byte-identical. The actual worker timeout was exercised; no detector evaluation or live product execution occurred.

## Next gates (proposed, not counted)

1. Reference evaluation: reopen the sealed reference checkpoint and test a matched synthetic validation comparison; the test split stays reserved. Do not claim downstream detector improvement.
2. Transfer and execution authority: verify an actual downstream encoder adapter and an explicit, scoped launch approval before connecting an executable route to the workbench.
3. Data: time annotation import/export and dataset verification on fixed licensed workloads; account for sample counts and recheck split isolation. Add caching only after measuring a bottleneck and without weakening freshness or authorization.
4. Iteration: compare errors at a fixed operating policy, require human approval for changes, preserve failures and never use training completion as release authority.
