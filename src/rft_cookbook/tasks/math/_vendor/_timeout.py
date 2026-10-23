# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for ai-runtime-rft-cookbook (2026): reduced from verl at commit 8718ca30
# (verl/utils/py_functional.py, `timeout_limit`, multiprocessing path only).
"""Run a function in a forked child with a time limit; sympy can hang on adversarial input."""

import multiprocessing
import queue
from functools import wraps


def _target(func, q, args, kwargs):
    try:
        q.put((True, func(*args, **kwargs)))
    except Exception as e:  # noqa: BLE001 - re-raised in the parent
        q.put((False, e))


def fork_timeout(seconds: float):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = multiprocessing.get_context("fork")
            q = context.Queue(maxsize=1)
            process = context.Process(target=_target, args=(func, q, args, kwargs))
            process.start()
            process.join(timeout=seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                raise TimeoutError(f"{func.__name__} timed out after {seconds} s")
            try:
                ok, value = q.get(timeout=0.1)
            except queue.Empty as err:
                raise RuntimeError(f"{func.__name__}: child exited with {process.exitcode} and no result") from err
            finally:
                q.close()
                q.join_thread()
            if ok:
                return value
            raise value

        return wrapper

    return decorator
