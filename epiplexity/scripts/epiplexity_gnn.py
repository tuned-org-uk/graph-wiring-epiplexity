from epiplexity.engine import EpiplexityEngine
from epiplexity.algorithms.epiplexity_gnn import GNNModelAdapter, GNNEpiplexityConfig

# model: GNN; dataset: list of (graph_obj, label_int)
adapter = GNNModelAdapter(model, GNNEpiplexityConfig())
engine = EpiplexityEngine(adapter, dataset)
engine.report()