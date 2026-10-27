# Third-party notices

New code in this repository is under the repository license ([LICENSE.md](LICENSE.md)). The files
below are derived from Apache-2.0 sources: they keep the Apache License 2.0
([LICENSE-APACHE](LICENSE-APACHE)), each carries a header naming its origin and whether it was
modified, and the original copyright notices are retained. This file records attribution; it is
not legal advice and does not establish what any model or dataset may be used for.

## Files derived from other projects

verl-on-air ([github.com/mshtelma/verl-on-air](https://github.com/mshtelma/verl-on-air)) is the earlier
open-source prototype (Apache-2.0) that this cookbook grew out of.

| file | origin | license |
|---|---|---|
| `src/rft_cookbook/tasks/math/_vendor/prm800k_grader.py` | `grade_answer` from verl (commit 8718ca30) `verl/utils/reward_score/prime_math/__init__.py` (PRIME team), itself from OpenAI's PRM800K grader | Apache-2.0 (verl, PRIME); PRM800K: MIT |
| `src/rft_cookbook/tasks/math/_vendor/math_normalize.py` | verl (commit 8718ca30) `prime_math/math_normalize.py`, from the MATH release's answer normalization (Hendrycks et al., 2021) | Apache-2.0; MIT (MATH) |
| `src/rft_cookbook/tasks/math/_vendor/_timeout.py` | verl (commit 8718ca30) `verl/utils/py_functional.py`, `timeout_limit` | Apache-2.0 |
| `src/rft_cookbook/tasks/math/{grading,calculator,data}.py`, `tests/test_math_{grading,calculator}.py` | verl-on-air `usecases/math/` | Apache-2.0 |
| `src/rft_cookbook/tasks/officeqa/answer_match.py` | the OfficeQA benchmark's `reward.py` (github.com/databricks/officeqa), by way of verl-on-air | Apache-2.0 |
| `src/rft_cookbook/tasks/officeqa/{answer_check,path_report,scoring,corpus,chunker,html_table_parser,data,calibration}.py`, `tools.json`, `data.lock.json` | verl-on-air `usecases/officeqa/` | Apache-2.0 |
| `src/rft_cookbook/tasks/search/{answer,retrieval,data}.py`, `heldout.ids` | verl-on-air `usecases/agentic-search/` (`reward.py`, `tool.py`, `prep_data.py`, `make_splits.py`, `splits/musique_test.ids`) | Apache-2.0 |
| `src/rft_cookbook/tools/{sandbox,sandbox_child}.py` | verl-on-air `usecases/officeqa/sandbox.py` | Apache-2.0 |
| `src/rft_cookbook/air/stage_model.py` | verl-on-air `engine/stage_model.py` | Apache-2.0 |
| `frameworks/verl/image/*` | verl-on-air `docker/` | Apache-2.0 |
| `frameworks/miles/rft_miles/launchers/qwen3_6_35b_a3b.py` | miles (radixark/miles, commit 8e99947) `scripts/run_qwen3_6_35b_a3b_mtp.py`, with its GLM-5.3-Flash launcher's options (`scripts/run_glm5_3_flash.py`); copyright RadixArk, Inc. | Apache-2.0 |

The last-`\boxed{}` extraction in `tasks/math/grading.py` follows the MATH benchmark's evaluation
(Hendrycks et al., 2021; MIT).

## Used, not copied

| what | used for | license |
|---|---|---|
| verl (commit 8718ca30) | the RL framework of the verl adapter (from its image only) | Apache-2.0 |
| vLLM, Megatron-Core, Megatron-Bridge, Transformer Engine, apex, flash-attn, PyTorch, Ray, Transformers | the training and inference stack in the verl image (`frameworks/verl/image/*.lock`) | each project's own (Apache-2.0, BSD-3-Clause, MIT) |
| Databricks AI Runtime base image | the verl and miles images' base | Databricks' terms |
| miles (radixark/miles, commit 8e99947; Slime by Zhipu AI) | the RL framework of the miles adapter (fetched at job start, `rft_miles/miles.lock.json`) | Apache-2.0 |
| SGLang, Megatron-LM, TorchTitan, bm25s, databricks-sdk | the stack in the miles image (`frameworks/miles/image/`) | each project's own (Apache-2.0, BSD-3-Clause, MIT) |

## Models and datasets (downloaded at a pinned revision; not redistributed)

| what | used for | where to read its license |
|---|---|---|
| Qwen3.5 (2B and larger), Qwen3.6-35B-A3B | policies | the model cards on Hugging Face |
| GLM-5.3-Flash | policy; hosted judge | the model cards on Hugging Face |
| MATH (`DigitalLearningGmbH/MATH-lighteval`; mirror `EleutherAI/hendrycks_math`) | math training | the MATH release (MIT) |
| MATH-500 (`HuggingFaceH4/MATH-500`) | math evaluation | a subset of MATH from OpenAI's PRM800K (MIT) |
| OfficeQA (`databricks/officeqa`) | OfficeQA questions and corpus | the OfficeQA release (code Apache-2.0; data CC BY-SA, see its LICENSE-CC-BY-SA) |

The OfficeQA test fixture (`tests/fixtures/officeqa_episode.json`) uses synthetic table text, not
benchmark content.

## Upstream NOTICE files

Apache-2.0 section 4(d) carries the NOTICE text of projects whose files are derived here.

verl ([Notice.txt](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/Notice.txt)):

```
Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
```

miles ([NOTICE](https://github.com/radixark/miles/blob/8e99947cc950791b82c74e5392370cb43df70637/NOTICE)):

```
Miles
Copyright 2025-2026 RadixArk, Inc.
This product includes software developed from Slime (https://github.com/THUDM/slime),
Copyright 2025 Zhipu AI, licensed under the Apache License, Version 2.0.
```
