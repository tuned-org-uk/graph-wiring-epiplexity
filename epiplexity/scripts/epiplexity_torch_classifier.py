from epiplexity.engine import EpiplexityEngine
from epiplexity.algorithms.torch_classifier import TorchClassifierModelAdapter, TorchClassifierEpiplexityConfig

# model: nn.Module; dataset: list of (x_tensor, y_int)
adapter = TorchClassifierModelAdapter(model, TorchClassifierEpiplexityConfig())
engine = EpiplexityEngine(adapter, dataset)
engine.report()