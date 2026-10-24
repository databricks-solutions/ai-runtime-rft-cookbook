"""rft_miles.diagnostics on CPU: families by Megatron name, and drift, last update and cosine."""
import pytest

torch = pytest.importorskip("torch")

from rft_miles import diagnostics  # noqa: E402


@pytest.mark.parametrize("name,family", [
    ("module.decoder.layers.3.mlp.router.weight", "router"),
    ("module.decoder.layers.3.mlp.experts.linear_fc1.weight0", "routed_experts"),
    ("module.decoder.layers.3.mlp.experts.linear_fc2.weight7", None),          # not sampled
    ("module.decoder.layers.3.mlp.shared_experts.gate_weight", "shared_experts"),
    ("module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight", "norms"),
    ("module.decoder.layers.3.self_attention.linear_attn.A_log", "linear_attention"),
    ("module.decoder.layers.3.self_attention.linear_proj.weight", "attention"),
    ("module.decoder.final_layernorm.weight", "norms"),
    ("module.module.embedding.word_embeddings.weight", "embedding"),
    ("module.module.output_layer.weight", "output"),
])
def test_a_parameter_belongs_to_its_family(name, family):
    assert diagnostics.family(name) == family


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(4, 4, bias=False)            # "router.weight": not ".mlp.router."
        self.register_parameter("w", torch.nn.Parameter(torch.ones(8)))

    def named_parameters(self, *args, **kwargs):
        yield "decoder.layers.0.mlp.router.weight", self.router.weight
        yield "decoder.layers.0.self_attention.linear_proj.weight", self.w


def test_drift_last_update_and_cosine(monkeypatch, capsys):
    monkeypatch.setattr(diagnostics, "_state", None)
    model = [Toy()]
    diagnostics.before_train_step(None, 0, 0, model, None, None)     # keeps the start
    assert "kept the starting weights" in capsys.readouterr().out
    with torch.no_grad():
        model[0].w += 0.1                                             # two updates in one direction
    diagnostics.before_train_step(None, 1, 0, model, None, None)
    with torch.no_grad():
        model[0].w += 0.1
    numbers = diagnostics.measure(model, diagnostics._state)
    attention = numbers["attention"]
    assert attention["drift"] == pytest.approx(0.2, rel=1e-5) and attention["last"] == pytest.approx(0.1, rel=1e-5)
    assert attention["cos"] == pytest.approx(1.0)
    assert numbers["router"]["drift"] == 0.0 and numbers["router"]["cos"] == 0.0
    with torch.no_grad():
        model[0].w -= 0.1                                             # one update back toward the start
    attention = diagnostics.measure(model, diagnostics._state)["attention"]
    assert attention["drift"] == pytest.approx(0.1, rel=1e-5) and attention["cos"] == pytest.approx(-1.0)
    printed = capsys.readouterr().out
    assert "[rft-diag] before update 1 attention: drift 1.00e-01" in printed and "[rft-diag-json]" in printed
