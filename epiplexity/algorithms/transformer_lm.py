# epiplexity_transformer_lm.py

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from epiplexity.model import TTimeProbabilisticModel


@dataclass
class TransformerLMEpiplexityConfig:
    bits_per_param: int = 16         # quantised LM weights if desired
    overhead_bits: float = 32_768.0  # ~4 KB for architecture + tokenizer
    max_length: int = 2048           # truncate long sequences for log_prob
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class TransformerLMModelAdapter(TTimeProbabilisticModel):
    """
    T-time probabilistic model wrapper for an auto-regressive transformer LM.

    Dataset items are raw strings or token id lists; we define P(text)
    via the product of token probabilities under the LM.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: TransformerLMEpiplexityConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self._num_params = sum(p.numel() for p in self.model.parameters())

        # Default BOS/EOS if available
        if self.config.bos_token_id is None and tokenizer.bos_token_id is not None:
            self.config.bos_token_id = tokenizer.bos_token_id
        if self.config.eos_token_id is None and tokenizer.eos_token_id is not None:
            self.config.eos_token_id = tokenizer.eos_token_id

    def description_length_bits(self) -> float:
        """
        S_T(X): bits to specify the LM weights + tokenizer.
        """
        param_bits = self._num_params * self.config.bits_per_param
        return float(param_bits + self.config.overhead_bits)

    def _to_token_ids(self, x) -> List[int]:
        """
        Convert a dataset item to a list of token ids.
        x can be a string or a list[int].
        """
        if isinstance(x, str):
            # Use tokenizer to encode; no special tokens so we can add BOS/EOS manually.
            ids = self.tokenizer.encode(x, add_special_tokens=False)
        else:
            ids = list(x)
        # Optionally wrap with BOS/EOS
        if self.config.bos_token_id is not None:
            ids = [self.config.bos_token_id] + ids
        if self.config.eos_token_id is not None:
            ids = ids + [self.config.eos_token_id]
        # Truncate for bounded compute
        if len(ids) > self.config.max_length:
            ids = ids[: self.config.max_length]
        return ids

    def log_prob(self, x) -> float:
        """
        log P(x) under the LM, using the standard teacher-forced likelihood.
        """
        token_ids = self._to_token_ids(x)
        if len(token_ids) < 2:
            return 0.0  # degenerate

        input_ids = torch.tensor([token_ids[:-1]], device=self.device)  # prefix
        target_ids = torch.tensor([token_ids[1:]], device=self.device)  # next tokens

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # (1, T-1, vocab_size)
            # Align logits with targets and compute log-softmax
            log_probs = torch.log_softmax(logits, dim=-1)
            # Gather log P(target_t | prefix_<t)
            lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            total_log_prob = lp.sum().item()
        return float(total_log_prob)

    def sample(self, n: int = 1, max_length: Optional[int] = None) -> List[List[int]]:
        """
        Simple ancestral sampling from the LM. Returns token id sequences.
        """
        max_len = max_length or self.config.max_length
        sequences = []

        for _ in range(n):
            cur_ids: List[int] = []
            if self.config.bos_token_id is not None:
                cur_ids.append(self.config.bos_token_id)

            for _step in range(max_len):
                input_ids = torch.tensor([cur_ids], device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids).logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                cur_ids.append(next_id)
                if self.config.eos_token_id is not None and next_id == self.config.eos_token_id:
                    break
            sequences.append(cur_ids)
        return sequences

    def raw_bits(self, X: Sequence) -> float:
        """
        Baseline: character-level 8-bit encoding of raw text,
        or 32 bits per token id if dataset items are already tokenised.
        """
        bits = 0.0
        for x in X:
            if isinstance(x, str):
                bits += 8.0 * len(x)
            else:
                # Assume 32-bit ints per token id
                bits += 32.0 * len(x)
        return float(bits)