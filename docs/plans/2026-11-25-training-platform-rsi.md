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

## Next gates (proposed, not counted)

1. Recipe design: separate supervised-from-scratch, supervised warm-start and self-supervised pretraining. Specify data/holdouts, objective, checkpoint provenance and baseline. An unimplemented recipe must fail closed before scheduling.
2. Data: measure annotation import/export and dataset build times on fixed licensed workloads; verify split isolation and per-class availability. Add caching only when one benchmark identifies a bottleneck.
3. Training adapters: prove one tiny local acceptance per permitted route before considering GPU/distributed support. Record hardware, time/storage budget and recovery.
4. Evaluation and iteration: compare frozen holdouts and errors at a fixed operating policy, with human approval for changes. Preserve failures and never use a training completion event as release authority.
