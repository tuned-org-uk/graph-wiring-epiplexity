# epiplexity_gnn.py

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from epiplexity.model import TTimeProbabilisticModel
from epiplexity.engine import EpiplexityEngine


@dataclass
class GNNEpiplexityConfig:
    bits_per_param: int = 32
    overhead_bits: float = 16_384.0     # architecture, message-passing scheme
    bits_per_node_feature: int = 32     # float32
    bits_per_edge_index: int = 32       # int32
    bits_per_graph_feature: int = 32    # optional global features


class GNNModelAdapter(TTimeProbabilisticModel):
    """
    T-time probabilistic model wrapper for a graph neural network classifier.

    We assume:
      - model(batch) -> logits over classes.
      - dataset items: (graph_obj, label_int), where graph_obj has:
            - x: node features tensor [num_nodes, F]
            - edge_index: [2, num_edges] tensor of int64
            - optional graph-level features (attr 'u' or 'graph_attr').
    """

    def __init__(
        self,
        model: nn.Module,
        config: GNNEpiplexityConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self._num_params = sum(p.numel() for p in self.model.parameters())

    def description_length_bits(self) -> float:
        """
        S_T(X): bits to specify GNN weights plus architecture.
        """
        param_bits = self._num_params * self.config.bits_per_param
        return float(param_bits + self.config.overhead_bits)

    def _to_batch(self, graph_obj: Any) -> Any:
        """
        Wrap a single graph object into a batch of size 1.

        For PyTorch Geometric, you would do: Batch.from_data_list([graph_obj]).
        Here we just assume the model can already consume a single graph object.
        """
        return graph_obj

    def log_prob(self, graph_label: Tuple[Any, int]) -> float:
        """
        log P(y | G) for a single (graph, label) pair.
        """
        graph, label = graph_label
        y = int(label)
        batch = self._to_batch(graph)

        # Move tensors to device if graph is a torch-geometric Data object
        if hasattr(batch, "to"):
            batch = batch.to(self.device)

        with torch.no_grad():
            logits = self.model(batch)  # shape (1, C) or (C,)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            log_probs = torch.log_softmax(logits, dim=-1)
            lp = log_probs[0, y].item()
        return float(lp)

    def sample(self, n: int = 1) -> List[Any]:
        """
        Sampling graphs from a discriminative GNN is non-trivial and usually
        not needed for epiplexity diagnostics.
        """
        raise NotImplementedError("Sampling graphs is task-specific.")

    def raw_bits(self, X: Sequence[Tuple[Any, int]]) -> float:
        """
        Baseline uncompressed size of graphs in bits.

        For each graph:
          - node features: num_nodes * F * bits_per_node_feature
          - edge_index: 2 * num_edges * bits_per_edge_index
          - optional graph-level features: num_features * bits_per_graph_feature
        """
        total_bits = 0.0
        for graph, _ in X:
            # Node features
            if hasattr(graph, "x") and graph.x is not None:
                num_floats = graph.x.numel()
                total_bits += num_floats * self.config.bits_per_node_feature
            # Edge indices
            if hasattr(graph, "edge_index") and graph.edge_index is not None:
                num_ints = graph.edge_index.numel()
                total_bits += num_ints * self.config.bits_per_edge_index
            # Optional graph-level features (PyG 'u' or 'graph_attr')
            if hasattr(graph, "u") and graph.u is not None:
                total_bits += graph.u.numel() * self.config.bits_per_graph_feature
            elif hasattr(graph, "graph_attr") and graph.graph_attr is not None:
                total_bits += graph.graph_attr.numel() * self.config.bits_per_graph_feature

        return float(total_bits)