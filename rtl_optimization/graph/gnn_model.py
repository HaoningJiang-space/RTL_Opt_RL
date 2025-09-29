#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL优化专用图神经网络模型
GNN model specialized for RTL optimization

该模块实现专门针对RTL网表优化的图神经网络模型，包括节点嵌入、
图卷积、全局池化和多任务输出头。
This module implements GNN model specialized for RTL netlist optimization,
including node embedding, graph convolution, global pooling and multi-task output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import logging

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.data import Data, Batch
except ImportError:
    # 如果PyTorch Geometric未安装，提供替代实现
    # Provide alternative implementation if PyTorch Geometric is not installed
    print("Warning: PyTorch Geometric not installed. Using simplified implementation.")

    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            return self.lin(x)

    def global_mean_pool(x, batch):
        if batch is None:
            return torch.mean(x, dim=0, keepdim=True)
        return torch.mean(x, dim=0, keepdim=True)

from ..utils.config import GNNConfig


class RTLOptimizationGNN(nn.Module):
    """RTL优化专用图神经网络 / GNN specialized for RTL optimization

    该模型专门针对RTL网表设计，能够学习电路结构特征并预测PPA指标，
    同时为强化学习提供图级嵌入表示。
    This model is specifically designed for RTL netlists, capable of learning
    circuit structure features and predicting PPA metrics, while providing
    graph-level embeddings for reinforcement learning.
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        """初始化GNN模型 / Initialize GNN model

        Args:
            config: GNN配置对象 / GNN configuration object
        """
        super(RTLOptimizationGNN, self).__init__()
        self.config = config or GNNConfig()
        self.logger = logging.getLogger(__name__)

        # 输入特征维度 / Input feature dimension
        self.input_dim = 24  # 根据RTL图表示的节点特征维度

        # 节点类型嵌入层 / Node type embedding layer
        self.node_type_embedding = nn.Embedding(4, 16)  # 4种节点类型

        # 门类型嵌入层 / Gate type embedding layer
        self.gate_type_embedding = nn.Embedding(9, 16)  # 8种门类型 + 1个unknown

        # 特征编码器 / Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim + 32, self.config.hidden_dim),  # 24 + 16 + 16
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate)
        )

        # 图卷积层 / Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(self.config.num_conv_layers):
            if i == 0:
                self.conv_layers.append(GCNConv(self.config.hidden_dim, self.config.hidden_dim))
            else:
                self.conv_layers.append(GCNConv(self.config.hidden_dim, self.config.hidden_dim))

        # 批归一化层 / Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.config.hidden_dim) for _ in range(self.config.num_conv_layers)
        ])

        # 注意力层用于重要节点加权 / Attention layer for important node weighting
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 图级池化 / Graph-level pooling
        self.pooling_method = "attention_mean"  # "mean", "max", "add", "attention_mean"

        # 图级特征处理 / Graph-level feature processing
        self.graph_processor = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # 多任务输出头 / Multi-task output heads

        # PPA预测头 / PPA prediction head
        self.ppa_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2, self.config.ppa_output_dim),
            nn.Sigmoid()  # 归一化到[0,1]
        )

        # RL值函数头 / RL value function head
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2, self.config.value_output_dim)
        )

        # 置信度预测头 / Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 初始化参数 / Initialize parameters
        self._initialize_weights()

        self.logger.info(f"RTL优化GNN模型初始化完成 / RTL optimization GNN model initialized: "
                        f"隐藏维度={self.config.hidden_dim}, "
                        f"卷积层数={self.config.num_conv_layers}")

    def _initialize_weights(self) -> None:
        """初始化模型权重 / Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, data: Any, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播 / Forward propagation

        Args:
            data: 图数据对象 (PyTorch Geometric Data或Batch对象)
                 Graph data object (PyTorch Geometric Data or Batch object)
            return_attention: 是否返回注意力权重 / Whether to return attention weights

        Returns:
            Dict[str, torch.Tensor]: 包含各种输出的字典
                                   Dictionary containing various outputs
        """
        # 提取图数据 / Extract graph data
        x, edge_index, batch = self._extract_graph_data(data)

        # 节点特征编码 / Node feature encoding
        x_encoded = self._encode_node_features(x)

        # 图卷积 / Graph convolution
        x_conv, attention_weights = self._apply_graph_convolution(x_encoded, edge_index, return_attention)

        # 图级池化 / Graph-level pooling
        graph_embedding = self._pool_graph_features(x_conv, batch, attention_weights)

        # 图级特征处理 / Graph-level feature processing
        graph_features = self.graph_processor(graph_embedding)

        # 多任务输出 / Multi-task outputs
        outputs = {
            "graph_embedding": graph_features,
            "ppa_prediction": self.ppa_predictor(graph_features),
            "state_value": self.value_head(graph_features),
            "confidence": self.confidence_head(graph_features)
        }

        if return_attention:
            outputs["attention_weights"] = attention_weights

        return outputs

    def _extract_graph_data(self, data: Any) -> tuple:
        """提取图数据 / Extract graph data"""
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # PyTorch Geometric Data对象
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, 'batch', None)
        else:
            # 假设是字典格式
            x = data.get('x', data.get('node_features'))
            edge_index = data.get('edge_index', data.get('edges'))
            batch = data.get('batch', None)

        return x, edge_index, batch

    def _encode_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """编码节点特征 / Encode node features

        Args:
            x: 原始节点特征 [num_nodes, feature_dim]

        Returns:
            torch.Tensor: 编码后的节点特征
        """
        batch_size, feature_dim = x.shape

        # 提取节点类型 (前4维是one-hot编码)
        # Extract node type (first 4 dimensions are one-hot encoded)
        node_type_indices = torch.argmax(x[:, :4], dim=1)
        node_type_emb = self.node_type_embedding(node_type_indices)

        # 提取门类型 (5-12维是门类型的one-hot编码)
        # Extract gate type (dimensions 5-12 are gate type one-hot encoding)
        gate_type_indices = torch.argmax(x[:, 4:12], dim=1)
        gate_type_emb = self.gate_type_embedding(gate_type_indices)

        # 组合所有特征 / Combine all features
        combined_features = torch.cat([x, node_type_emb, gate_type_emb], dim=1)

        # 特征编码 / Feature encoding
        return self.feature_encoder(combined_features)

    def _apply_graph_convolution(self, x: torch.Tensor, edge_index: torch.Tensor,
                                return_attention: bool) -> tuple:
        """应用图卷积 / Apply graph convolution"""
        attention_weights = None

        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # 图卷积 / Graph convolution
            x = conv(x, edge_index)

            # 批归一化 / Batch normalization
            x = bn(x)

            # 激活函数 / Activation function
            if self.config.activation == "relu":
                x = F.relu(x)
            elif self.config.activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif self.config.activation == "gelu":
                x = F.gelu(x)

            # Dropout
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)

        # 计算注意力权重 / Compute attention weights
        if return_attention or self.pooling_method == "attention_mean":
            attention_weights = self.attention(x)

        return x, attention_weights

    def _pool_graph_features(self, x: torch.Tensor, batch: Optional[torch.Tensor],
                           attention_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """图级特征池化 / Graph-level feature pooling"""

        if batch is None:
            # 单个图的情况 / Single graph case
            if self.pooling_method == "mean":
                return torch.mean(x, dim=0, keepdim=True)
            elif self.pooling_method == "max":
                return torch.max(x, dim=0, keepdim=True)[0]
            elif self.pooling_method == "add":
                return torch.sum(x, dim=0, keepdim=True)
            elif self.pooling_method == "attention_mean":
                if attention_weights is not None:
                    weighted_x = x * attention_weights
                    return torch.sum(weighted_x, dim=0, keepdim=True) / torch.sum(attention_weights)
                else:
                    return torch.mean(x, dim=0, keepdim=True)
        else:
            # 批量图的情况 / Batch graphs case
            try:
                if self.pooling_method == "mean":
                    return global_mean_pool(x, batch)
                elif self.pooling_method == "max":
                    return global_max_pool(x, batch)
                elif self.pooling_method == "add":
                    return global_add_pool(x, batch)
                elif self.pooling_method == "attention_mean":
                    if attention_weights is not None:
                        weighted_x = x * attention_weights
                        pooled = global_add_pool(weighted_x, batch)
                        weights_sum = global_add_pool(attention_weights, batch)
                        return pooled / (weights_sum + 1e-8)  # 避免除零
                    else:
                        return global_mean_pool(x, batch)
            except:
                # 降级方案 / Fallback
                return torch.mean(x, dim=0, keepdim=True)

        return torch.mean(x, dim=0, keepdim=True)

    def predict_ppa(self, data: Any) -> Dict[str, torch.Tensor]:
        """预测PPA指标 / Predict PPA metrics

        Args:
            data: 图数据 / Graph data

        Returns:
            Dict[str, torch.Tensor]: PPA预测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)

            # 解析PPA预测 / Parse PPA prediction
            ppa_pred = outputs["ppa_prediction"]

            return {
                "delay": ppa_pred[:, 0],
                "area": ppa_pred[:, 1],
                "power": ppa_pred[:, 2],
                "confidence": outputs["confidence"]
            }

    def get_graph_embedding(self, data: Any) -> torch.Tensor:
        """获取图嵌入表示 / Get graph embedding representation

        Args:
            data: 图数据 / Graph data

        Returns:
            torch.Tensor: 图嵌入向量
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            return outputs["graph_embedding"]

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失 / Compute multi-task loss

        Args:
            outputs: 模型输出 / Model outputs
            targets: 目标值 / Target values

        Returns:
            Dict[str, torch.Tensor]: 各种损失
        """
        losses = {}

        # PPA预测损失 / PPA prediction loss
        if "ppa_target" in targets:
            ppa_loss = F.mse_loss(outputs["ppa_prediction"], targets["ppa_target"])
            losses["ppa_loss"] = ppa_loss

        # 值函数损失 / Value function loss
        if "value_target" in targets:
            value_loss = F.mse_loss(outputs["state_value"], targets["value_target"])
            losses["value_loss"] = value_loss

        # 置信度损失 / Confidence loss
        if "confidence_target" in targets:
            confidence_loss = F.binary_cross_entropy(
                outputs["confidence"], targets["confidence_target"]
            )
            losses["confidence_loss"] = confidence_loss

        # 总损失 / Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses

    def save_model(self, filepath: str) -> None:
        """保存模型 / Save model"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "model_class": self.__class__.__name__
        }, filepath)
        self.logger.info(f"模型已保存 / Model saved: {filepath}")

    @classmethod
    def load_model(cls, filepath: str, device: str = "cpu") -> "RTLOptimizationGNN":
        """加载模型 / Load model"""
        checkpoint = torch.load(filepath, map_location=device)

        config = checkpoint.get("config", GNNConfig())
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        logging.getLogger(__name__).info(f"模型已加载 / Model loaded: {filepath}")
        return model

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息 / Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_class": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "hidden_dim": self.config.hidden_dim,
            "num_conv_layers": self.config.num_conv_layers,
            "dropout_rate": self.config.dropout_rate,
            "activation": self.config.activation,
            "pooling_method": self.pooling_method
        }


class RTLGraphEncoder(nn.Module):
    """RTL图编码器 / RTL graph encoder

    专门用于将RTL图编码为固定维度的向量表示，
    可以作为其他模型的组件使用。
    Specifically for encoding RTL graphs into fixed-dimension vector representations,
    can be used as a component in other models.
    """

    def __init__(self, output_dim: int = 128, config: Optional[GNNConfig] = None):
        """初始化图编码器 / Initialize graph encoder

        Args:
            output_dim: 输出维度 / Output dimension
            config: GNN配置 / GNN configuration
        """
        super(RTLGraphEncoder, self).__init__()
        self.output_dim = output_dim
        self.gnn = RTLOptimizationGNN(config)

        # 降维层 / Dimension reduction layer
        self.projection = nn.Linear(self.gnn.config.hidden_dim, output_dim)

    def forward(self, data: Any) -> torch.Tensor:
        """前向传播 / Forward propagation"""
        with torch.no_grad():
            gnn_outputs = self.gnn(data)
            graph_embedding = gnn_outputs["graph_embedding"]

        # 投影到目标维度 / Project to target dimension
        return self.projection(graph_embedding)


# 工具函数 / Utility functions

def create_rtl_gnn_model(config: Optional[GNNConfig] = None,
                        pretrained_path: Optional[str] = None,
                        device: str = "cpu") -> RTLOptimizationGNN:
    """创建RTL GNN模型 / Create RTL GNN model

    Args:
        config: 模型配置 / Model configuration
        pretrained_path: 预训练模型路径 / Pretrained model path
        device: 设备 / Device

    Returns:
        RTLOptimizationGNN: 初始化的模型
    """
    if pretrained_path and os.path.exists(pretrained_path):
        model = RTLOptimizationGNN.load_model(pretrained_path, device)
    else:
        model = RTLOptimizationGNN(config)

    model.to(device)
    return model


# 测试代码 / Test code
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 测试模型 / Test model
    config = GNNConfig()
    model = RTLOptimizationGNN(config)

    print("RTL优化GNN模型测试 / RTL optimization GNN model test:")
    print(f"模型信息 / Model info: {model.get_model_info()}")

    # 创建测试数据 / Create test data
    num_nodes = 10
    x = torch.randn(num_nodes, 24)  # 节点特征
    edge_index = torch.randint(0, num_nodes, (2, 20))  # 边索引

    # 简单的测试数据结构 / Simple test data structure
    test_data = {
        'x': x,
        'edge_index': edge_index
    }

    # 前向传播测试 / Forward propagation test
    outputs = model(test_data)
    print(f"输出形状 / Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    print("GNN模型测试完成 / GNN model test completed")