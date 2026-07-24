"""Extract residual-stream activations from a white-box model for the linear probe.

The seam between a loaded HuggingFace causal LM (the ``hf`` backend / the released Qwen
organisms) and ``probes.linear.ProbeExample``: run ONE forward pass over a prompt with
``output_hidden_states=True`` and read the residual stream at a chosen token position for
a set of layers. That vector is the probe's input.

Two backends, one contract (mirrors the rest of the package):
  * a REAL transformers ``CausalLM`` (a ``torch.nn.Module``) — the Colab organism run;
    torch is used ONLY here, on that path.
  * a lightweight FAKE exposing ``.hidden_states`` — the tests, which need no torch/GPU.

``hidden_states`` from a real model is a tuple of length ``n_layers + 1`` (the embedding
output plus each block), each shaped ``[batch, seq, d_model]``; index 0 is the embeddings
and index ``L`` is the residual stream after block ``L``. We squeeze the batch dim and
read row ``position`` (default ``-1``: the last prompt token — the natural probe site for
a trigger context, and the site the additional-computation-trace measures).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .linear import ProbeExample


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _tokenize(tokenizer, prompt: str) -> list[int]:
    """Token ids for ``prompt`` as a flat python list (real HF tokenizer or a fake dict)."""
    enc = tokenizer(prompt)
    ids = enc["input_ids"] if _get(enc, "input_ids") is not None else enc
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):  # a batched [[...]] encoding
        ids = list(ids[0])
    return [int(x) for x in ids]


def _forward_hidden(model, input_ids: Sequence[int], device: str) -> list[np.ndarray]:
    """Per-layer hidden states as numpy arrays shaped ``[seq, d]``.

    Real path (torch ``nn.Module``): a single ``output_hidden_states=True`` forward pass
    under ``no_grad``. Fake path: ``model(input_ids)`` returns an object/dict whose
    ``hidden_states`` is a sequence of ``[seq, d]`` or ``[1, seq, d]`` arrays.
    """
    try:
        import torch

        is_module = isinstance(model, torch.nn.Module)
    except ImportError:
        is_module = False

    if is_module:
        import torch

        with torch.no_grad():
            t = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
            out = model(input_ids=t, output_hidden_states=True)
        return [np.asarray(h[0].detach().to("cpu"), dtype=float) for h in out.hidden_states]

    out = model(input_ids)
    hs = _get(out, "hidden_states")
    if hs is None:
        raise ValueError("fake model returned no hidden_states")
    layers: list[np.ndarray] = []
    for h in hs:
        a = np.asarray(h, dtype=float)
        if a.ndim == 3:  # [1, seq, d] -> [seq, d]
            a = a[0]
        layers.append(a)
    return layers


def residual_stream(
    model,
    tokenizer,
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    position: int = -1,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """The residual-stream vector at ``position`` for each requested layer: ``{layer: vec[d]}``.

    ``position`` is a token index (default ``-1`` = the last token). ``layers`` selects
    hidden_states indices (default: all). Every prompt scored by the same model yields the
    same layer set and per-layer dimension, so the resulting ProbeExamples stack cleanly.
    """
    ids = _tokenize(tokenizer, prompt)
    if not ids:
        raise ValueError("empty tokenization for prompt")
    hs = _forward_hidden(model, ids, device)
    n = len(hs)
    if layers is None:
        layers = list(range(n))
    seq_len = int(hs[0].shape[0])
    pos = position if position >= 0 else seq_len + position
    if not (0 <= pos < seq_len):
        raise IndexError(f"position {position} out of range for seq_len {seq_len}")
    return {int(l): np.asarray(hs[l][pos], dtype=float) for l in layers}


def _forward_logits(model, input_ids: Sequence[int], device: str) -> np.ndarray:
    """Per-position next-token logits as numpy ``[seq, vocab]`` (real torch or fake)."""
    try:
        import torch

        is_module = isinstance(model, torch.nn.Module)
    except ImportError:
        is_module = False
    if is_module:
        import torch

        with torch.no_grad():
            t = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
            out = model(input_ids=t)
        return np.asarray(out.logits[0].detach().to("cpu"), dtype=float)
    out = model(input_ids)
    logits = np.asarray(_get(out, "logits"), dtype=float)
    if logits.ndim == 3:  # [1, seq, vocab] -> [seq, vocab]
        logits = logits[0]
    return logits


def _log_softmax_at(row: np.ndarray, index: int) -> float:
    m = float(np.max(row))
    return float(row[index] - m - np.log(np.sum(np.exp(row - m))))


def _render_context(messages: Sequence[Message]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def make_hf_scorer(model, tokenizer, *, device: str = "cpu"):
    """A ``score_fn(context_messages, completion) -> summed logprob`` for the logprob trace.

    Computes the EXACT summed log-probability the model assigns to ``completion`` given the
    rendered context, in ONE forward pass — it does NOT route through the hf adapter's
    generation shell. Real path uses torch; a fake model (``.logits``) drives the tests.
    Returns None if either side tokenizes empty.
    """
    def score(context_messages: Sequence[Message], completion: str) -> float | None:
        ctx_ids = _tokenize(tokenizer, _render_context(context_messages))
        comp_ids = _tokenize(tokenizer, completion)
        if not ctx_ids or not comp_ids:
            return None
        logits = _forward_logits(model, list(ctx_ids) + list(comp_ids), device)
        total = 0.0
        for j, tid in enumerate(comp_ids):
            pos = len(ctx_ids) + j - 1          # the position whose logits PREDICT comp token j
            total += _log_softmax_at(logits[pos], int(tid))
        return total

    return score


def build_probe_examples(
    model,
    tokenizer,
    labeled: Iterable[Mapping[str, Any]],
    *,
    layers: Sequence[int] | None = None,
    position: int = -1,
    device: str = "cpu",
    default_level: str = "L5",
) -> list[ProbeExample]:
    """Map labeled prompt rows -> ``ProbeExample`` via the residual stream at ``position``.

    Each row is a mapping with keys ``organism_id``, ``is_loyal``, ``prompt`` and
    optionally ``principal``, ``group`` (the matched-pair leakage unit — default the
    organism_id), and ``level`` (affordance tag). This is the ONLY function the Day-1
    Colab driver calls to turn provided organisms + affordance-graded prompts into probe
    inputs; everything downstream (``select_layer`` / ``crossval_fixed_fpr`` /
    ``transfer_matrix``) is already built and tested.
    """
    examples: list[ProbeExample] = []
    for row in labeled:
        acts = residual_stream(model, tokenizer, row["prompt"], layers=layers, position=position, device=device)
        examples.append(
            ProbeExample(
                organism_id=row["organism_id"],
                is_loyal=bool(row["is_loyal"]),
                activations=acts,
                principal=row.get("principal"),
                level=row.get("level", default_level),
                group=row.get("group"),
            )
        )
    return examples
