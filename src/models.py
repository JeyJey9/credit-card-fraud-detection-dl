"""
Multi-Layer Perceptron (MLP) architecture for Credit Card Fraud Detection.
Supports configurable hidden layers, Batch Normalization, Dropout, activation functions,
and prior probability bias initialization (Lin et al., 2017 [31]).
"""

import math
from typing import List
import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
}


class FraudMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron for tabular binary classification under extreme class imbalance.
    Each hidden block consists of:
        Linear(in_dim, out_dim) -> BatchNorm1d(out_dim) -> Activation() -> Dropout(p)
    Followed by a linear output head producing raw logits.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: List[int] = (128, 64, 32),
        dropout: float = 0.3,
        activation: str = "relu",
        use_batchnorm: bool = True,
        prior_prob: float = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.activation_name = activation.lower()
        self.use_batchnorm = use_batchnorm

        act_cls = ACTIVATIONS.get(self.activation_name, nn.ReLU)

        layers = []
        in_features = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 1)

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float = None) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Focal Loss prior initialization for output layer bias: b = -log((1 - pi) / pi)
        if prior_prob is not None and 0.0 < prior_prob < 1.0:
            bias_val = -math.log((1.0 - prior_prob) / prior_prob)
            nn.init.constant_(self.classifier.bias, bias_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fraud probabilities in [0, 1] for input tensor x."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
