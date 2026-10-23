"""The miles adapter on CPU: the resolved launcher arguments, the agent function against a scripted
chat endpoint, the reward hook, in-place checkpoint publishing, and the checked-in jobs."""
from __future__ import annotations

import asyncio
import os
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rft_cookbook.agent.harness import ByteCounter, ChatTurn
from rft_cookbook.air.jobs import JOB_ENV, SNAPSHOT_PATHS, data_key
from rft_cookbook.checkpoints import CheckpointStore
from rft_cookbook.recipe import load_recipe, recipes_root
from rft_miles import agent, checkpoint, reward
from rft_miles.render import GPUS_PER_NODE, LAUNCHER, resolve_run, stage_job

SITE = {"volume_root": "/Volumes/cat/sch/vol/rft"}
RECIPE = "math/miles/glm-5.3-flash-4layer_b300x8"
FULL = "math/miles/glm-5.3-flash_b300x32"
QWEN = "officeqa/miles/qwen3.6-35b-a3b_h100x16"
ROOT = recipes_root()


def resolve(tmp_path, overrides=()):
    out = resolve_run(RECIPE, site=SITE, run_id="glm-test-01", overrides=list(overrides), out=tmp_path / "run")
    return out, json.loads((out / "plan.json").read_text()), json.loads((out / "miles.json").read_text())


def _flags(args):
    """--flag value pairs; a flag without a value (--no-...) maps to True."""
    flags, i = {}, 0
    while i < len(args):
        valued = i + 1 < len(args) and not args[i + 1].startswith("--")
        flags[args[i]] = args[i + 1] if valued else True
        i += 2 if valued else 1
    return flags


def test_the_cookbook_hooks_ride_on_upstreams_launcher(tmp_path):
    out, plan, launcher = resolve(tmp_path)
    assert launcher["script"] == "scripts/run_glm5_3_flash.py"
    options = dict(zip(launcher["options"][::2], launcher["options"][1::2]))
    assert options["--model-name"] == "GLM-5.3-Flash-4layer" and options["--num-gpus-per-node"] == "8"
    assert options["--num-rollout"] == "2" and options["--rollout-max-response-len"] == "1024"
    extra = launcher["extra_args"]
    flags = _flags(extra)
    assert flags["--custom-generate-function-path"] == "miles.rollout.generate_hub.agentic_tool_call.generate"
    assert flags["--custom-agent-function-path"] == "rft_miles.agent.run" and flags["--use-session-server"] == "v2"
    assert flags["--session-message-matcher"] == "rft_miles.session.matches" and "--tito-model" not in flags
    assert flags["--custom-agent-function-mode"] == "inline"   # the episode sink outlives an episode
    assert flags["--save-debug-event-data"].startswith(plan["local_dir"])   # appended to: not on the Volume
    assert flags["--no-log-inference-engine-weight-checksums"] is True
    assert flags["--custom-rm-path"] == "rft_miles.reward.reward"
    assert flags["--custom-megatron-post-save-hook-path"] == "rft_miles.checkpoint.publish"
    assert flags["--save"] == flags["--load"] == plan["checkpoints"] == "/Volumes/cat/sch/vol/rft/runs/glm-test-01/checkpoints"
    assert flags["--use-checkpoint-opt-param-scheduler"] is True   # a resume to more rollouts loads
    assert flags["--sglang-tool-call-parser"] == "glm47" and flags["--rollout-batch-size"] == "4"
    assert flags["--sglang-sampling-defaults"] == "openai"
    assert shlex.split(shlex.join(extra)) == extra   # survives upstream's --extra-args string
    # the math quickstart's prepared data serves this recipe too
    quickstart = load_recipe("math/verl/qwen3.5-2b_h100x8")
    assert plan["data_dir"].rsplit("/", 1)[1] == data_key(quickstart, plan["task_version"])
    assert plan["megatron_dir"] == plan["model_dir"] + "-megatron-" + plan["miles_commit"][:12]


@pytest.mark.parametrize("override,message", [("compute.nodes=0", "at least 1"),
                                              ("train.save_interval=0", "save_interval"),
                                              ("model.tool_parser=null", "model.tool_parser"),
                                              ("model.processor_from.revision=main", "processor_from")])
def test_inconsistent_recipes_do_not_resolve(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        resolve(tmp_path, [override])


def test_the_full_model_trains_on_four_nodes_in_upstreams_layout(tmp_path):
    out = resolve_run(FULL, site=SITE, run_id="glm-full-01", overrides=[], out=tmp_path / "full")
    plan, launcher = json.loads((out / "plan.json").read_text()), json.loads((out / "miles.json").read_text())
    options = dict(zip(launcher["options"][::2], launcher["options"][1::2]))
    assert options["--model-name"] == "GLM-5.3-Flash" and options["--num-nodes"] == "4"
    assert options["--num-gpus-per-node"] == "8"   # upstream's launcher accepts 32 or 64 GPUs for this model
    flags = _flags(launcher["extra_args"])
    assert flags["--save-interval"] == "3" and flags["--n-samples-per-prompt"] == "8"
    assert plan["processor_dir"] is None   # the full model ships its processor


def test_the_qwen_twin_trains_and_evaluates_as_its_verl_recipe_does(tmp_path):
    """The miles recipe of a model with a verl recipe matches it wherever miles allows."""
    from rft_miles.render import launcher_path

    out = resolve_run(QWEN, site=SITE, run_id="oqa-test-01", out=tmp_path / "run",
                      overrides=["eval.baseline_run=oqa-verl-01", "stages=[prep,stage_model,stage_judge,calibrate,"
                                 "train,export,eval,compare]"])
    plan, launcher = json.loads((out / "plan.json").read_text()), json.loads((out / "miles.json").read_text())
    recipe = yaml.safe_load((out / "recipe.yaml").read_text())
    twin = load_recipe("officeqa/verl/qwen3.6-35b-a3b_h100x16")
    for section in ("data", "agent", "scoring"):
        assert recipe[section] == twin[section], section
    assert {k: v for k, v in recipe["eval"].items() if k not in ("accelerator", "timeout_minutes", "baseline_run")} == \
        {k: v for k, v in twin["eval"].items() if k != "baseline_run"} | {"seed": None}
    assert (recipe["model"]["id"], recipe["model"]["revision"]) == (twin["model"]["id"], twin["model"]["revision"])
    assert plan["baseline_dir"] == "/Volumes/cat/sch/vol/rft/runs/oqa-verl-01/eval/baseline"   # the verl run's
    assert plan["data_dir"] == plan["data_dir"].replace("oqa-test-01", "") and "officeqa" in plan["data_dir"]
    t, v = recipe["train"], twin["train"]
    assert t["num_rollout"] == v["total_training_steps"] and t["rollout_batch_size"] == v["ppo_mini_batch_size"]
    assert t["n_samples_per_prompt"] == v["rollout_n"]
    assert (t["lr"], t["kl_loss_coef"], t["seed"]) == (v["lr"], v["kl_loss_coef"], v["seed"])
    flags = _flags(launcher["extra_args"])
    assert flags["--global-batch-size"] == "16" and flags["--kl-loss-coef"] == "0.01" and flags["--lr"] == "1e-06"
    assert flags["--rollout-max-prompt-len"] == "2048" and flags["--rollout-max-context-len"] == "98304"
    assert flags["--tito-model"] == "qwen36"   # Qwen3.6's appendable template and token boundary
    assert flags["--save-hf"] == f"{plan['checkpoints']}/iter_{{rollout_id:07d}}/hf"
    assert launcher_path(recipe, Path("/miles")).name == "qwen3_6_35b_a3b.py"   # the cookbook's own launcher
    assert launcher_path(load_recipe(FULL), Path("/miles")) == Path("/miles/scripts/run_glm5_3_flash.py")
    assert plan["abort_file"] == f"{plan['run_dir']}/train/ABORT.json"
    with pytest.raises(ValueError, match="baseline_eval"):   # a borrowed baseline is not evaluated again
        resolve_run(QWEN, site=SITE, run_id="oqa-test-02", out=tmp_path / "run2", overrides=["eval.baseline_run=x-01"])


def test_the_cookbook_launcher_trains_no_mtp_and_matches_verls_algorithm(monkeypatch):
    """Against a stand-in for miles' command_utils: the train arguments the launcher builds."""
    import dataclasses
    import importlib.util
    import sys
    import types

    from rft_miles.render import launcher_path

    stub = types.ModuleType("miles.utils.external_utils.command_utils")

    @dataclasses.dataclass
    class ExecuteTrainConfig:
        num_nodes: int = 1

    stub.ExecuteTrainConfig, stub.create_run_id, stub.dataclass_cli = ExecuteTrainConfig, lambda: "run", lambda f: f
    for name in ("miles", "miles.utils", "miles.utils.external_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.utils.external_utils.command_utils", stub)
    sys.modules["miles.utils.external_utils"].command_utils = stub
    monkeypatch.setitem(sys.modules, "typer", types.ModuleType("typer"))
    path = launcher_path(load_recipe(QWEN), Path("/miles"))
    spec = importlib.util.spec_from_file_location("rft_test_launcher", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    args = module.ScriptArgs(hf_checkpoint="/hf", ckpt_dir="/ckpt", extra_args="--lr 2e-6")
    train = module.train_args(args)
    assert "mtp" not in train.lower() and "speculative" not in train   # verl trains no MTP, decodes plainly
    flags = _flags(train.split())
    assert flags["--ref-load"] == "/ckpt/qwen3.6-35B-A3B_torch_dist" and flags["--hf-checkpoint"] == "/hf"
    assert (flags["--weight-decay"], flags["--adam-beta2"], flags["--eps-clip-c"]) == ("0.01", "0.999", "3.0")
    assert flags["--calculate-per-token-loss"] is True and flags["--moe-aux-loss-coeff"] == "0.001"
    assert (flags["--tensor-model-parallel-size"], flags["--context-parallel-size"],
            flags["--expert-model-parallel-size"]) == ("2", "4", "8")
    assert flags["--sglang-sampling-defaults"] == "openai" and flags["--rollout-top-k"] == "-1"
    # no FlashInfer TRT-LLM kernels to build at start-up (the image has no prebuilt copies)
    assert flags["--sglang-attention-backend"] == flags["--sglang-moe-runner-backend"] == "triton"
    assert flags["--log-probs-backend"] == "fused"   # no fp32 copy of the logits on an 80 GB GPU
    # the trainer's allocator grows its segments, as the shell that runs the command reads it
    (env,) = [shlex.split(part)[0] for part in train.split("--train-env-vars")[1:]]
    assert json.loads(env) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    # separate GPUs, as verl's separate_async: the first node trains, the second serves
    assert "--colocate" not in flags and "--offload-train" not in flags
    assert (flags["--actor-num-nodes"], flags["--actor-num-gpus-per-node"], flags["--rollout-num-gpus"]) == ("1", "8", "8")
    assert flags["--rollout-num-gpus-per-engine"] == "8"
    with pytest.raises(SystemExit, match="even number"):
        module.train_args(module.ScriptArgs(num_nodes=1))
    assert train.split()[-2:] == ["--lr", "2e-6"]   # the cookbook's --extra-args come last and win
    launched = {}
    module.ScriptArgs.create_backend = lambda self: SimpleNamespace(execute_train=lambda **kw: launched.update(kw))
    module.execute(args)
    assert launched["train_script"] == "train_async.py"   # the next rollout generates while one trains
    # every option the adapter passes is one of the launcher's
    out_options = {"model_name", "num_nodes", "num_gpus_per_node", "hf_checkpoint", "ckpt_dir", "num_rollout",
                   "rollout_max_response_len", "train_offload_dir", "run_id", "extra_args"}
    assert out_options <= {f.name for f in dataclasses.fields(module.ScriptArgs)}


def test_the_slice_gets_the_full_models_processor_on_its_local_copy(tmp_path):
    """SGLang loads this architecture's image processor, which the slice does not ship."""
    from rft_miles.launch import _add_processor_files

    out, plan, _ = resolve(tmp_path)
    assert plan["processor_dir"].endswith("/models/GLM-5.3-Flash@eb9eb208eb0d-tokenizer")   # the judge's tokenizer dir
    recipe = yaml.safe_load((out / "recipe.yaml").read_text())
    local = {"processor_dir": str(tmp_path / "processor"), "local_hf": str(tmp_path / "local")}
    for directory in local.values():
        Path(directory).mkdir()
    (tmp_path / "processor" / "processor_config.json").write_text("{}")
    _add_processor_files(recipe, local)
    assert (tmp_path / "local" / "processor_config.json").read_text() == "{}"


def test_the_agent_function_runs_and_scores_one_episode(tmp_path, monkeypatch):
    seen = []

    class ScriptedClient:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        async def complete(self, messages, tools, max_tokens):
            assert messages[0]["role"] == "system" and tools
            return ChatTurn(text="</think>\n\nThe sum is \\boxed{4}.")

    monkeypatch.setattr(agent, "OpenAIChatClient", ScriptedClient)
    monkeypatch.setattr(agent, "TemplateCounter", lambda tokenizer, kwargs: ByteCounter())
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: object()))
    agent._setup.cache_clear()
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"task": "math", "budgets": {"max_turns": 2}, "resources": None, "tokenizer": "tok",
                                  "episode_dir": str(tmp_path / "episodes"), "model": {"id": "m"},
                                  "tool_parser": "glm47", "served_model": "model", "request_timeout_s": 60}))
    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": "What is 2+2?"}],
           "ground_truth": "4", "metadata": {"level": 1}}
    result = asyncio.run(agent.run("http://router/sessions/s1", None, {"temperature": 0.8, "max_tokens": 1024},
                                   {agent.ROW_KEY: row, agent.CONFIG_KEY: str(config)}))
    assert result[agent.REWARD_KEY] == 1.0 and result["rft_scored"] is True
    assert seen[0]["base_url"] == "http://router/sessions/s1/v1" and seen[0]["temperature"] == 0.8
    sample = SimpleNamespace(metadata=dict(result), index=0)
    assert seen[0]["top_k"] == -1   # sent even when miles passes none: no server default applies
    assert asyncio.run(reward.reward(None, sample)) == 1.0
    # no score: no reward, and miles drops the group (apply_missing_reward_filter) as verl does
    assert asyncio.run(reward.reward(None, SimpleNamespace(metadata={}, index=1))) is None
    # a session's samples come together when the generate function returns several (session server v2)
    assert asyncio.run(reward.reward(None, [sample, SimpleNamespace(metadata={}, index=1)])) == [1.0, None]


def _strict(stored, new):
    """miles' strict_message_matches (message_matcher_hub), for a test without miles installed."""
    def value(message, key):
        v = message.get(key)
        v = None if v in (None, "", []) else v
        return [{k: x for k, x in c.items() if k != "index"} for c in v] if key == "tool_calls" and v else v
    return all(value(stored, k) == value(new, k) for k in ("role", "content", "reasoning_content", "tool_calls"))


def test_a_replayed_turn_continues_its_session_as_one_lineage(tmp_path, monkeypatch):
    """The harness replays a turn as it read it (reasoning split off, unanswered calls left out):
    miles' strict matcher would branch the session there, rft_miles.session accepts it."""
    import sys
    import types

    from rft_cookbook.agent.policy import ParsedCall
    from rft_miles import session

    try:
        from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches as strict
    except ImportError:
        hub = types.ModuleType("miles.utils.chat_template_utils.message_matcher_hub")
        hub.strict_message_matches = strict = _strict
        for name in ("miles", "miles.utils", "miles.utils.chat_template_utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, hub.__name__, hub)
    raw = "I add them.</think>\n\nTwo calls."
    stored = {"role": "assistant", "content": raw, "tool_calls": [   # SGLang's message, no reasoning parser
        {"id": "call_a", "index": 0, "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}},
        {"id": "call_b", "index": 1, "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2*2"}'}}]}
    replays = []

    class SessionServer:
        def __init__(self, **kwargs):
            pass

        async def complete(self, messages, tools, max_tokens):
            if len(messages) == 2:
                calls = [ParsedCall(c["function"]["name"], c["function"]["arguments"], c["id"]) for c in stored["tool_calls"]]
                return ChatTurn(text=raw, tool_calls=calls)
            replays.append(messages[2])
            return ChatTurn(text="</think>\n\nThe sum is \\boxed{4}.")

    monkeypatch.setattr(agent, "OpenAIChatClient", SessionServer)
    monkeypatch.setattr(agent, "TemplateCounter", lambda tokenizer, kwargs: ByteCounter())
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: object()))
    agent._setup.cache_clear()
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"task": "math", "budgets": {"max_turns": 3}, "resources": None, "tokenizer": "tok",
                                  "episode_dir": None, "model": {"id": "m"}, "tool_parser": "qwen3_coder",
                                  "served_model": "model", "request_timeout_s": 60}))
    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": "What is 2+2?"}],
           "ground_truth": "4", "metadata": {}}
    result = asyncio.run(agent.run("http://router/sessions/s1", None, {}, {agent.ROW_KEY: row, agent.CONFIG_KEY: str(config)}))
    assert result[agent.REWARD_KEY] == 1.0
    replayed, = replays
    assert replayed["reasoning_content"] == "I add them." and [c["id"] for c in replayed["tool_calls"]] == ["call_a"]
    assert not strict(stored, replayed) and session.matches(stored, replayed)
    # anything else the replay changes is another history
    assert not session.matches(stored, {**replayed, "content": "One call."})
    assert not session.matches(stored, {**replayed, "tool_calls": [{**replayed["tool_calls"][0], "id": "call_c"}]})
    assert not session.matches(stored, {**replayed, "tool_calls": stored["tool_calls"][::-1]})
    assert not session.matches({"role": "tool", "content": "4"}, {"role": "tool", "content": "4.0"})
    # a server that splits the reasoning itself is replayed verbatim
    split = {"role": "assistant", "content": "Two calls.", "reasoning_content": "I add them."}
    assert session.matches(split, dict(split))


def test_an_excluded_episode_has_no_reward_and_a_run_of_them_stops_training(tmp_path, monkeypatch):
    class BrokenServer:
        def __init__(self, **kwargs):
            pass

        async def complete(self, messages, tools, max_tokens):
            raise ConnectionError("engine gone")

    monkeypatch.setattr(agent, "OpenAIChatClient", BrokenServer)
    monkeypatch.setattr(agent, "TemplateCounter", lambda tokenizer, kwargs: ByteCounter())
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: object()))
    agent._setup.cache_clear()
    monkeypatch.setattr(agent, "_FAILED_IN_A_ROW", [0])
    abort = tmp_path / "train" / "ABORT.json"
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"task": "math", "budgets": {"max_turns": 2}, "resources": None, "tokenizer": "tok",
                                  "episode_dir": None, "model": {"id": "m"}, "tool_parser": "glm47",
                                  "served_model": "model", "request_timeout_s": 60,
                                  "abort_file": str(abort), "max_consecutive_exclusions": 2}))
    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": "What is 2+2?"}],
           "ground_truth": "4", "metadata": {}}
    metadata = {agent.ROW_KEY: row, agent.CONFIG_KEY: str(config)}
    first = asyncio.run(agent.run("http://router/sessions/s1", None, {}, metadata))
    assert first[agent.REWARD_KEY] is None and first["rft_excluded"] == "infra_failure" and not abort.exists()
    asyncio.run(agent.run("http://router/sessions/s2", None, {}, metadata))
    assert "2 episodes in a row were excluded" in json.loads(abort.read_text())["reason"]


def test_an_agent_function_that_cannot_run_counts_as_excluded(tmp_path, monkeypatch):
    def broken(path):
        raise OSError("no tokenizer at tok")

    monkeypatch.setattr(agent, "_setup", broken)
    monkeypatch.setattr(agent, "_FAILED_IN_A_ROW", [0])
    abort = tmp_path / "train" / "ABORT.json"
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"abort_file": str(abort), "max_consecutive_exclusions": 2}))
    metadata = {agent.ROW_KEY: {"id": "q1"}, agent.CONFIG_KEY: str(config)}
    for session in ("s1", "s2"):
        with pytest.raises(OSError):
            asyncio.run(agent.run(f"http://router/sessions/{session}", None, {}, metadata))
    assert "last: OSError: no tokenizer at tok" in json.loads(abort.read_text())["reason"]


def test_a_megatron_save_is_published_in_place(tmp_path, capsys):
    save = tmp_path / "checkpoints"
    iteration = save / "iter_0000001"
    iteration.mkdir(parents=True)
    for name, size in (("__0_0.distcp", 64), ("common.pt", 8), (".metadata", 4), ("metadata.json", 2)):
        (iteration / name).write_bytes(b"x" * size)
    (save / "latest_checkpointed_iteration.txt").write_text("1")
    checkpoint.publish(SimpleNamespace(save=str(save)), 1, str(iteration), None)
    latest = CheckpointStore(save).latest()
    assert (latest.step, latest.name, latest.location) == (2, "iter_0000001", str(iteration))
    assert {f["path"] for f in latest.manifest["files"]} == {"__0_0.distcp", "common.pt", ".metadata", "metadata.json"}
    assert "published iter_0000001 (step 2)" in capsys.readouterr().out
    unfinished = save / "iter_0000003"
    unfinished.mkdir()
    with pytest.raises(RuntimeError, match="completion marker"):
        checkpoint.publish(SimpleNamespace(save=str(save)), 3, str(unfinished), None)


@pytest.mark.parametrize("ref", sorted(str(p.parent.relative_to(recipes_root()))
                                       for p in recipes_root().glob("*/miles/*/recipe.yaml")))
def test_the_checked_in_jobs_match_the_recipe(ref):
    recipe = load_recipe(ref)
    jobs = ROOT / ref / "jobs"
    assert sorted(p.stem for p in jobs.glob("*.yaml")) == sorted(recipe["stages"])
    for stage in recipe["stages"]:
        path = jobs / f"{stage}.yaml"
        job, expected = yaml.safe_load(path.read_text()), stage_job(recipe, stage)
        assert job["command"] == f"bash ${{CODE_SOURCE_PATH}}/{LAUNCHER} {stage}"
        env = job["env_variables"]
        assert env["RFT_RECIPE"] == ref and all(env[name].startswith("<") for name in JOB_ENV)
        assert job["environment"]["docker_image"]["url"] == expected["image"], stage
        snapshot = job["code_source"]["snapshot"]
        assert (path.parent / snapshot["root_path"]).resolve() == ROOT.parent
        assert snapshot["include_paths"] == SNAPSHOT_PATHS
        assert job["compute"] == {"num_accelerators": expected["gpus"], "accelerator_type": expected["accelerator"]}, stage
        assert job["timeout_minutes"] == expected["timeout_minutes"] and job["max_retries"] == 0, stage
        # B300 nodes have no /local_disk0: their scratch moves to /tmp; H100 nodes keep it there
        if expected["accelerator"] == "GPU_8xB300":
            assert env.get("RFT_LOCAL_ROOT") == "/tmp/rft", stage
        elif expected["accelerator"] == "GPU_8xH100":
            assert "RFT_LOCAL_ROOT" not in env, stage


def test_a_stage_job_takes_the_compute_its_stage_needs():
    recipe = load_recipe(QWEN)
    train, stage, evaluate, judge = (stage_job(recipe, s) for s in ("train", "stage_model", "eval", "stage_judge"))
    assert (train["gpus"], train["accelerator"], train["image"]) == (16, "GPU_8xH100", "<miles-image>")
    assert train["timeout_minutes"] == 1080 and stage["timeout_minutes"] == 240
    assert (evaluate["gpus"], evaluate["accelerator"], evaluate["image"]) == (8, "GPU_8xH100", "<verl-image>")
    assert (judge["gpus"], judge["accelerator"], judge["timeout_minutes"]) == (1, "GPU_1xA10", 60)
    full = load_recipe(FULL)
    assert stage_job(full, "train")["gpus"] == 32 and stage_job(full, "stage_model")["gpus"] == 8


def test_on_several_nodes_node_0_runs_miles_on_the_cluster_and_the_others_join(tmp_path, monkeypatch):
    from rft_cookbook.air import cluster
    from rft_miles import launch

    out = resolve_run(FULL, site={"volume_root": str(tmp_path / "vol")}, run_id="glm-full-02", overrides=[],
                      out=tmp_path / "run")
    recipe, paths = launch._load(out)
    calls = []
    monkeypatch.setattr(launch, "_copy_dir", lambda source, target: calls.append(("copy", target)))
    monkeypatch.setattr(launch, "_agent_inputs", lambda recipe, paths: 7)
    monkeypatch.setattr(launch, "_run_miles", lambda *args: calls.append(("miles", os.environ.get("MILES_SCRIPT_EXTERNAL_RAY"))))
    monkeypatch.setattr(cluster, "run_head", lambda nodes, gpus, rendezvous, body: (calls.append(("head", gpus)), body()))
    monkeypatch.setattr(cluster, "run_worker", lambda nodes, gpus, rendezvous: calls.append(("worker", nodes.rank)))
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("NUM_NODES", "2")
    with pytest.raises(SystemExit, match="trains on 4 nodes, the job has 2"):
        launch.train(recipe, paths, out)
    monkeypatch.setenv("NUM_NODES", "4")
    monkeypatch.setenv("POD_RANK", "2")
    launch.train(recipe, paths, out)
    assert calls == [("copy", paths["local_hf"]), ("copy", paths["local_megatron"]), ("worker", 2)]
    calls.clear()
    monkeypatch.setenv("POD_RANK", "0")
    launch.train(recipe, paths, out)
    assert calls[-2:] == [("head", 8), ("miles", "1")]   # miles submits to the cookbook's Ray cluster
