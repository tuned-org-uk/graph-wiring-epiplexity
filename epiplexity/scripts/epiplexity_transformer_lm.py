from transformers import AutoModelForCausalLM, AutoTokenizer
from epiplexity.engine import EpiplexityEngine
from epiplexity.algorithms.transformer_lm import TransformerLMModelAdapter, TransformerLMEpiplexityConfig

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

dataset = [
    "The quick brown fox jumps over the lazy dog.",
    "A transformer language model defines a distribution over token sequences.",
]

adapter = TransformerLMModelAdapter(model, tokenizer, TransformerLMEpiplexityConfig())
engine = EpiplexityEngine(adapter, dataset)
engine.report()