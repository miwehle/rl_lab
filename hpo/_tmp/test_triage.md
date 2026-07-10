# HPO Test Triage

Purpose: classify HPO tests before simplifying them according to `../nmt_lab/translator/how_to_test.md` and the HPO public API rules.

## Rules

- `keep`: directly tests `notebook-public` or `to-higher-level public` behavior with acceptable coupling.
- `rewrite`: tests valuable public behavior, but currently reaches through internals, patches private helpers, or asserts too many implementation details.
- `drop`: directly tests private helpers or incidental implementation details without enough independent behavior value.
- `drop` also applies when a test is fragile: a harmless refactor, renaming, layout adjustment, or implementation-local change would likely break it even though public behavior still works.

Simplification target: production behavior coverage should remain useful, but test-code LOC should go down. Avoid adding helper infrastructure unless it clearly removes more complexity than it adds.

Priority means how strongly a test module deviates from the test rules, not only which file to edit first.

- High priority: the module is clearly questionable under `how_to_test.md`; it likely tests too many internals, is fragile, too detailed, too large, or tightly coupled to layout/implementation details.
- Medium priority: the module is mostly useful, but has some smells or simplification opportunities.
- Low priority: the module is relatively clean and mostly tests public behavior; leave it alone unless touched for another reason.

## High-Priority Files

### `hpo/tests/notebook/test_dashboard.py` (436 LOC)

Overall: `rewrite` with several `drop` candidates.
Action: rewrite/drop layout-detail assertions.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| ------- | ------------- | ------- | ---------- | --------- | ------------ |
| ==hi==  | med           | med     | med        | ==hi==    | med          |

Hi-smell details:

- Fragile: many assertions would fail after a harmless dashboard layout refresh even if the dashboard still communicates the same information.
- Overspec.: the tests pin exact Plotly layout numbers, annotation positions, axis choices, and legend details instead of only the user-visible dashboard behavior.

Action details:

- `keep`: reporter-level behavior through `Dashboard`, such as rejecting unknown render modes, preserving optimization/training/robustness context, throttling training updates, and showing stored checkpoint robustness.
- `rewrite`: plot semantic tests should assert durable user-facing facts, such as evaluation checkpoint score usage, empty-state existence, live trial params, robustness summaries, and training progress content.
- `drop`: exact layout numbers and brittle Plotly internals, such as figure width/height, subplot domains, margin values, legend coordinates, annotation indexes, exact axis object choices, and hidden-axis implementation details.

### `hpo/tests/evaluation/test_lander_rendering.py` (83 LOC)

Overall: mixed `keep` and `drop`.
Action: drop private-helper tests first.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| ==hi== | ==hi== | lo | lo | med | lo |

Hi-smell details:

- Fragile: private overlay formatting and direction helpers can change during rendering refactors while public rendering behavior remains valid.
- Private Impl.: `_overlay_lines` and `_kick_direction` are tested directly even though they are private implementation helpers.

Action details:

- `keep`: public rendering behavior through `LanderRenderWrapper` and `world_colors`, including custom colors, score accumulation/reset, and rejection of unknown worlds.
- `rewrite`: overlay behavior if it matters to notebook/video users should be checked through public rendering or a public overlay API, not by calling private formatting helpers.
- `drop`: direct tests of `_overlay_lines` and `_kick_direction`; they are private implementation details and fragile against formatting or rendering refactors.

### `hpo/tests/test_study.py` (318 LOC)

Overall: mostly `keep`, but several tests should be `rewrite`.
Action: rewrite private loader/creator seam tests.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | ==hi== | med | med | med | med |

Hi-smell details:

- Private Impl.: many tests patch `_create_or_load_study`, `_load_study`, `_create_study`, or call `_create_or_load_study` directly instead of testing through public orchestration.

Action details:

- `keep`: `StudyRunner`, `Baseline`, and `run_study` public behavior: incumbent handling, finished-study skip, baseline loading, study attrs, progress reporting, and target trial semantics.
- `rewrite`: tests that patch `_create_or_load_study`, `_load_study`, `_create_study`, or call `_create_or_load_study` directly. The behavior is valuable, but the test shape is coupled to private orchestration seams.
- `drop` candidate: `test_create_or_load_study_records_runtime_metadata` if the same metadata behavior is covered through a public `StudyRunner` or `run_study` path.

### `hpo/tests/evaluation/test_video.py` (192 LOC)

Overall: mostly `keep` with a few `rewrite/drop` candidates.
Action: keep public video behavior, drop or rewrite private progress/hold details.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | med | med | med | med | lo |

Action details:

- `keep`: public video API behavior: one greedy episode is recorded, world/seed product is generated, color/overlay options are honored, invalid colors are rejected, and displayed conditions are formatted.
- `rewrite`: terminal-frame hold behavior should be tested only if it is a user-visible video guarantee; otherwise it is implementation detail.
- `drop`: direct patching of `_FINAL_HOLD_FRAMES` and `_tqdm` if progress/hold behavior is not part of the public contract.

## Medium-Priority Files

### `hpo/tests/evaluation/test_checkpoint_robustness.py` (171 LOC)

Overall: mostly `keep`.
Action: keep, reconsider progress-bar internals.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | med | med | med | med | lo |

Action details:

- `keep`: public evaluation behavior for `evaluate_checkpoint_robustness`, `checkpoint_scores`, `robustness_over_all_worlds`, and `score_summary`.
- `rewrite/drop`: `_tqdm` patching for progress bars is a fragile implementation-detail test unless progress reporting is considered a stable user-facing behavior.

### `hpo/tests/test_checkpointing.py` (231 LOC)

Overall: mostly `keep`, with a naming/structure cleanup opportunity.
Action: keep behavior, simplify structure only when LOC goes down.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | lo | med | med | med | med |

Action details:

- `keep`: checkpoint recorders, `ObjectiveHookFactory`, `best_checkpoint`, `save_checkpoint`, and `load_checkpoint` are public or core package contracts.
- `rewrite`: group class behavior in `Test...` classes and reduce repeated context setup if doing so lowers LOC.
- `watch`: some hook tests inspect attrs and paths closely; keep only attrs that are part of the persisted checkpoint contract.

### `hpo/tests/test_objective.py` (325 LOC)

Overall: mostly `keep`.
Action: keep, reduce fixture boilerplate cautiously.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | lo | ==hi== | ==hi== | med | med/hi |

Hi-smell details:

- Obscure: several tests require substantial FakeTrainer/FakeHooks/FakeEnv setup before the tested behavior becomes visible.
- Code Dupl.: fake trials, trainers, hooks, and environment factories repeat similar protocol scaffolding across objective scenarios.

Action details:

- `keep`: `ObjectiveConfig`, `create_objective`, and `evaluate_greedy_q_net` are notebook-public/developer-public.
- `rewrite`: if possible, reduce FakeTrainer/FakeHooks boilerplate and group class/function tests symmetrically.
- `watch`: tests that assert exact internal training config defaults are useful only for defaults that are intentionally part of the public objective contract.

### `hpo/tests/evaluation/test_hp_robustness.py` (158 LOC)

Overall: mostly `keep`.
Action: keep, move reporting DTO test later if symmetry is improved.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | lo | med | med | med | med |

Action details:

- `keep`: `select_robust_best` is a public robustness helper and the tests cover meaningful selection/reporting behavior.
- `rewrite`: consider whether `RobustnessProgress` belongs in `test_study_reporting.py` once test symmetry is improved.

### `hpo/tests/notebook/test_plots.py` (54 LOC)

Overall: mostly `keep`, with a small fragility watch.
Action: keep, avoid exact plotting internals unless user-visible.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | lo | lo | lo | med | lo |

Action details:

- `keep`: notebook plotting helpers are public for exploratory analysis.
- `rewrite/drop`: assertions about exact Matplotlib internals or locator classes should stay only when they protect a real notebook usability contract.

### `hpo/tests/notebook/test_drive_backup.py` (39 LOC)

Overall: mostly `keep`, one `rewrite` candidate.
Action: keep, rewrite private backup failure seam if simple.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | med | lo | lo | med | lo |

Action details:

- `keep`: public backup/restore behavior.
- `rewrite`: direct patching of `_backup_sqlite` is private; prefer public failure behavior if it can be induced simply.

## Low-Priority / Mostly Fine Files

### `hpo/tests/solar_system_lander/test_environment.py` (100 LOC)

Overall: mostly `keep`.
Action: keep, decide status of `acceleration_vector`.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | med | lo | lo | med | lo |

Action details:

- `keep`: `EnvFactory`, `World`, `DEFAULT_WORLD_MIX`, and observation modes are public environment behavior.
- `watch`: `acceleration_vector` is not currently notebook-public via `hpo`, but it is a named lower-level function used as domain logic. Either make it `to-higher-level public` if direct tests should remain, or test it through `EnvWrapper` behavior.

### `hpo/tests/solar_system_lander/test_reward_shaping.py` (34 LOC)

Overall: `keep`.
Action: keep.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | lo | lo | lo | lo | lo |

Action details:

- `keep`: `GroundThrustPenaltyEnv` is notebook-public and tests are already grouped in `TestGroundThrustPenaltyEnv`.

### `hpo/tests/lunar_lander/test_environment.py` (8 LOC)

Overall: `keep`.
Action: keep.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | lo | lo | lo | lo | lo |

Action details:

- `keep`: `EnvFactory` is public in `hpo.lunar_lander`.

### `hpo/tests/lunar_lander/test_logging.py` (73 LOC)

Overall: likely `keep`, but not urgent for notebook-public cleanup.
Action: keep unless logging becomes internal-only.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| med | lo | lo | lo | med | lo |

Action details:

- `keep`: file logging and `log_call` are public module behavior used by HPO infrastructure.
- `watch`: if logging is treated as internal infrastructure later, reduce tests to only the observable log contract needed by public HPO flows.

### `hpo/tests/notebook/test_colab.py` (55 LOC)

Overall: `keep`.
Action: keep.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | lo | lo | lo | lo | lo |

Action details:

- `keep`: `setup_colab`, `prepare_storage`, and `Storage.backup` are notebook-public helpers.

### `hpo/tests/notebook/test_optuna.py` (28 LOC)

Overall: `keep`.
Action: keep.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | lo | lo | lo | lo | lo |

Action details:

- `keep`: `db_path`, `db_summary`, and `neighbors` are notebook helpers.

### `hpo/tests/test_study_metadata.py` (46 LOC)

Overall: `keep`.
Action: keep.

| Fragile | Private Impl. | Obscure | Code Dupl. | Overspec. | Rewrite Risk |
| --- | --- | --- | --- | --- | --- |
| lo | lo | lo | lo | lo | lo |

Action details:

- `keep`: runtime metadata persistence is package infrastructure with observable database effects.
- `watch`: if this remains internal-only, keep tests focused on database contract rather than implementation helpers.

## Suggested First Pass

1. Start with `test_lander_rendering.py`: small file, obvious private-helper drops, low risk.
2. Then simplify `test_dashboard.py`: largest fragility payoff, but preserve semantic dashboard coverage.
3. Then revisit `test_study.py`: valuable behavior, but current monkeypatch seams are private and brittle.
