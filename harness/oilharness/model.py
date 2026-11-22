"""Typed readout over OpenTSLM-SoftPrompt. Spec rev 2 §4 (pass-2 N4: read logits at each sample's last real token)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import os as _os
OPENTSLM_SRC = Path(_os.environ.get("OPENTSLM_SRC", "../../OpenTSLM/src"))
if str(OPENTSLM_SRC) not in sys.path:
    sys.path.insert(0, str(OPENTSLM_SRC))

# The authors' package imports the Flamingo variant (open_flamingo) at package level; the SoftPrompt path
# never uses it, so a stub module keeps that optional dependency out of this harness.
import types as _types  # noqa: E402
if "open_flamingo" not in sys.modules:
    class _Placeholder:  # never instantiated on the SoftPrompt path
        pass
    _root = _types.ModuleType("open_flamingo"); _root.__path__ = []  # mark as a package
    _src = _types.ModuleType("open_flamingo.src"); _src.__path__ = []
    _lm = _types.ModuleType("open_flamingo.src.flamingo_lm"); _utils = _types.ModuleType("open_flamingo.src.utils")
    _root.Flamingo = _Placeholder
    _lm.FlamingoLayer = _Placeholder; _lm.FlamingoLMMixin = _Placeholder
    _utils.extend_instance = lambda *a, **k: None
    _root.src = _src; _src.flamingo_lm = _lm; _src.utils = _utils
    for _name, _mod in (("open_flamingo", _root), ("open_flamingo.src", _src),
                        ("open_flamingo.src.flamingo_lm", _lm), ("open_flamingo.src.utils", _utils)):
        sys.modules[_name] = _mod

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP  # noqa: E402

from .examples import LABEL_CANDIDATES, K_BINS  # noqa: E402


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def single_token_labels(tokenizer, k: int = K_BINS) -> tuple[list[str], list[int]]:
    """Choose k labels that each encode to exactly one token for this tokenizer; fail loud otherwise (spec §4)."""
    labels, ids = [], []
    for cand in LABEL_CANDIDATES + [f" {d}" for d in range(10)]:
        toks = tokenizer.encode(cand, add_special_tokens=False)
        if len(toks) == 1 and toks[0] not in ids:
            labels.append(cand); ids.append(toks[0])
        if len(labels) == k:
            return labels, ids
    raise RuntimeError(f"tokenizer {tokenizer.name_or_path}: could not find {k} single-token labels; found {labels}")


def check_tokenizers(llm_ids: list[str]) -> dict[str, list[str]]:
    """Startup check for every base named in the configs (pass-1 F7)."""
    from transformers import AutoTokenizer
    out = {}
    for llm_id in llm_ids:
        tok = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        out[llm_id] = single_token_labels(tok)[0]
    return out


class TypedOpenTSLM:
    """OpenTSLMSP plus a K-way readout: softmax over the label tokens' logits at the last prompt token."""

    def __init__(self, llm_id: str, device: str, lora: bool = True, lora_r: int = 16, lora_alpha: int = 32):
        self.device = device
        self.model = OpenTSLMSP(llm_id=llm_id, device=device)
        if lora:
            self.model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        self.labels, self.label_ids = single_token_labels(self.model.tokenizer)
        self.label_ids_t = torch.as_tensor(self.label_ids, device=device)
        self.temperature = 1.0

    # --- parameters ---
    def param_groups(self, lr_encoder: float, lr_projector: float, lr_lora: float, weight_decay: float):
        groups = [
            {"params": list(self.model.encoder.parameters()), "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": list(self.model.projector.parameters()), "lr": lr_projector, "weight_decay": weight_decay},
        ]
        lora = self.model.get_lora_parameters()
        if lora:
            groups.append({"params": lora, "lr": lr_lora, "weight_decay": weight_decay})
        return groups

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # --- readout ---
    def _decoder_and_head(self):
        """The inner decoder and the output head, through the PEFT wrapper when LoRA is on."""
        llm = self.model.llm
        causal = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        decoder = getattr(causal, "model", None); head = getattr(causal, "lm_head", None)
        if decoder is None or head is None:
            raise RuntimeError(f"{type(causal).__name__} has no .model/.lm_head; add a case for this architecture")
        return decoder, head

    def enable_gradient_checkpointing(self):
        decoder, _ = self._decoder_and_head()
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(decoder, "config"):
            decoder.config.use_cache = False

    def typed_logits(self, batch: list[dict]) -> torch.Tensor:
        """[B, K] logits of the K label tokens at each sample's last real position.

        Computed from the last hidden state and the K label rows of the output head only: never the full
        [B, L, V] logits (that was 20+ GB on a 0.5B model with a 150k vocabulary)."""
        inputs_embeds, attention_mask = self.model.pad_and_apply_batch(batch)
        decoder, head = self._decoder_and_head()
        out = decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state                               # [B, L, H]
        last = attention_mask.sum(dim=1) - 1                         # right padding: last real token per sample
        rows = torch.arange(hidden.size(0), device=hidden.device)
        h_last = hidden[rows, last]                                  # [B, H]
        w = head.weight.index_select(0, self.label_ids_t)            # [K, H]
        logits = h_last.float() @ w.float().T                        # [B, K]
        if getattr(head, "bias", None) is not None:
            logits = logits + head.bias.index_select(0, self.label_ids_t).float()
        return logits

    ordinal: bool = False   # set from cfg["ordinal"] by the runner; soft cross-entropy against the Gaussian-through-edges target

    def loss(self, batch: list[dict]) -> torch.Tensor:
        logits = self.typed_logits(batch)
        if self.ordinal and "soft" in batch[0]:
            soft = torch.as_tensor([b["soft"] for b in batch], device=logits.device, dtype=logits.dtype)
            return -(soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        target = torch.as_tensor([b["bin"] for b in batch], device=logits.device)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def probabilities(self, batch: list[dict]) -> torch.Tensor:
        return torch.softmax(self.typed_logits(batch) / self.temperature, dim=-1)

    def text_loss(self, batch: list[dict]) -> torch.Tensor:
        """The paper's own generation loss (config switch; spec §4)."""
        return self.model.compute_loss(batch)

    # --- checkpoints ---
    def save(self, path: str) -> None:
        self.model.store_to_file(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck["temperature"] = self.temperature
        ck["labels"] = self.labels
        torch.save(ck, path)

    def load(self, path: str) -> None:
        self.model.load_from_file(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.temperature = float(ck.get("temperature", 1.0))
