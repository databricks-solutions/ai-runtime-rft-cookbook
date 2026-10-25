"""How far each update moves the trainer's weights, by parameter family (`train.diagnostics: true`).

    --custom-megatron-before-train-step-hook-path rft_miles.diagnostics.before_train_step

miles calls the hook on every training rank before each update's forward and backward pass. The first
call keeps a host-memory copy of this process's starting weights (the routed experts only through
each layer's first local expert: all of them would not fit); every call then prints, on rank 0, one
line per family:

    [rft-diag] before update 12 router: drift 2.1e-04 last 1.9e-05 cos 0.84 (40 tensors)

`drift` is ||w - w0|| / ||w0|| since the process started (a resumed run starts again), `last` the
same for the latest update alone, and `cos` the cosine between the latest update and the drift: near 1
when the updates keep pushing one way, near 0 when they wander. Updates that keep one direction add
up linearly; wandering ones grow like the square root of their number. Weights replicated over
tensor- or context-parallel ranks count once per replica on both sides of each ratio, which leaves
the ratios unchanged. A JSON copy of each update's numbers follows on a `[rft-diag-json]` line.
"""
from __future__ import annotations

import json
import re

import torch

# first match wins; Megatron's names (miles' megatron_to_hf converters read the same ones)
FAMILIES = (
    ("router", re.compile(r"\.mlp\.router\.")),
    ("routed_experts", re.compile(r"\.mlp\.experts\.")),
    ("shared_experts", re.compile(r"\.shared_experts?\.")),
    ("norms", re.compile(r"layer_?norm|\.norm\.|_norm\.")),
    ("linear_attention", re.compile(r"\.linear_attn\.")),
    ("attention", re.compile(r"\.self_attention\.")),
    ("embedding", re.compile(r"embedding")),
    ("output", re.compile(r"output_layer")),
    ("other", re.compile(r"")),
)
NAMES = [name for name, _ in FAMILIES]
SAMPLED_EXPERT = re.compile(r"\.experts\..*weight0$")   # TEGroupedMLP keeps one tensor per local expert
CHUNK = 1 << 26   # elements per temporary: 256 MB in fp32

_state: dict | None = None


def family(name: str) -> str | None:
    """The parameter's family, or None for a routed expert that is not sampled."""
    for label, pattern in FAMILIES:
        if pattern.search(name):
            if label == "routed_experts" and not SAMPLED_EXPERT.search(name):
                return None
            return label
    return "other"


def _parameters(model):
    for chunk in model:
        for name, param in chunk.named_parameters():
            label = family(name)
            if label is not None:
                yield name, label, param


def _snapshot(model) -> dict:
    tensors = {name: (label, param, param.detach().to("cpu", copy=True)) for name, label, param in _parameters(model)}
    base = torch.zeros(len(NAMES), dtype=torch.float64)
    counts = torch.zeros(len(NAMES), dtype=torch.float64)
    for label, _, start in tensors.values():
        base[NAMES.index(label)] += start.double().pow(2).sum()
        counts[NAMES.index(label)] += 1
    return {"tensors": tensors, "previous": {name: start.clone() for name, (_, _, start) in tensors.items()},
            "base": _all_reduce(base), "counts": _all_reduce(counts)}


def _all_reduce(values: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else values.device
        values = values.to(device)
        torch.distributed.all_reduce(values)
    return values.cpu()


def measure(model, state: dict) -> dict[str, dict[str, float]]:
    """Each family's drift, latest update and their cosine; advances the previous copy to now."""
    sums = torch.zeros(3, len(NAMES), dtype=torch.float64)   # ||w - w0||^2, ||w - w_prev||^2, (w - w_prev).(w - w0)
    for name, (label, param, start) in state["tensors"].items():
        column, previous = NAMES.index(label), state["previous"][name]
        now, start_flat, previous_flat = param.detach().reshape(-1), start.reshape(-1), previous.reshape(-1)
        for lo in range(0, now.numel(), CHUNK):
            current = now[lo:lo + CHUNK].float()
            drift = current - start_flat[lo:lo + CHUNK].to(current.device, non_blocking=True).float()
            last = current - previous_flat[lo:lo + CHUNK].to(current.device, non_blocking=True).float()
            sums[0, column] += drift.double().pow(2).sum().item()
            sums[1, column] += last.double().pow(2).sum().item()
            sums[2, column] += (last.double() * drift.double()).sum().item()
        previous.copy_(param.detach())
    sums = _all_reduce(sums.reshape(-1)).reshape(3, len(NAMES))
    out = {}
    for column, label in enumerate(NAMES):
        if not state["counts"][column]:
            continue
        base = state["base"][column].sqrt().item() or 1.0
        drift, last = sums[0, column].sqrt().item(), sums[1, column].sqrt().item()
        cos = sums[2, column].item() / (drift * last) if drift and last else 0.0
        out[label] = {"drift": drift / base, "last": last / base, "cos": cos, "tensors": int(state["counts"][column])}
    return out


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
    global _state
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # the previous update's weights are in place
    if _state is None:
        _state = _snapshot(model)
        if _rank() == 0:
            print(f"[rft-diag] kept the starting weights at update {rollout_id}: "
                  + ", ".join(f"{label} {int(n)}" for label, n in zip(NAMES, _state["counts"].tolist()) if n)
                  + " tensors", flush=True)
            unnamed = [(name, tuple(start.shape), float(start.double().norm()))
                       for name, (label, _, start) in _state["tensors"].items() if label == "other"]
            if unnamed:   # what no family pattern names, with its starting size: a drift there needs a look
                print("[rft-diag] other on this rank: " + "; ".join(f"{n} {s} norm {v:.3g}" for n, s, v in unnamed[:8]),
                      flush=True)
        return
    numbers = measure(model, _state)
    if _rank() == 0:
        for label, n in numbers.items():
            print(f"[rft-diag] before update {rollout_id} {label}: drift {n['drift']:.2e} last {n['last']:.2e} "
                  f"cos {n['cos']:.2f} ({n['tensors']} tensors)", flush=True)
        print("[rft-diag-json] " + json.dumps({"update": int(rollout_id), "step": int(step_id), "families": numbers}),
              flush=True)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
