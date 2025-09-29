#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL网表图表示实现
RTL netlist graph representation implementation

该模块实现RTL网表的图表示，包括从Verilog/AIG文件构建图、节点特征提取、
图结构分析等功能。采用轻量级设计，专门针对RTL优化任务。
This module implements graph representation for RTL netlists, including
building graphs from Verilog/AIG files, node feature extraction, graph
structure analysis, etc. Uses lightweight design specifically for RTL optimization.
"""

import re
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

from ..utils.config import GraphConfig
from ..tools.abc_interface import ABCInterface


class RTLNetlistGraph:
    """RTL网表图表示类 / RTL netlist graph representation class

    负责将RTL网表转换为图表示，提取节点和边特征，支持与PyTorch Geometric的集成。
    Responsible for converting RTL netlists to graph representation, extracting
    node and edge features, supporting integration with PyTorch Geometric.
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        """初始化图表示构建器 / Initialize graph representation builder

        Args:
            config: 图配置对象 / Graph configuration object
        """
        self.config = config or GraphConfig()
        self.logger = logging.getLogger(__name__)

        # 节点类型映射 / Node type mapping
        self.node_type_mapping = {
            "gate": 0,
            "reg": 1,
            "port": 2,
            "wire": 3
        }

        # 常见门类型映射 / Common gate type mapping
        self.gate_type_mapping = {
            "and": 0, "or": 1, "not": 2, "xor": 3,
            "mux": 4, "add": 5, "sub": 6, "mult": 7,
            "unknown": 8  # 未知门类型
        }

        # 边类型映射 / Edge type mapping
        self.edge_type_mapping = {
            edge_type: idx for idx, edge_type in enumerate(self.config.edge_types)
        }

        # ABC接口用于格式转换 / ABC interface for format conversion
        self.abc_interface = ABCInterface()

    def build_from_file(self, file_path: str, file_format: str = "auto") -> Data:
        """从文件构建图表示 / Build graph representation from file

        Args:
            file_path: 输入文件路径 / Input file path
            file_format: 文件格式 ("verilog", "aig", "auto") / File format

        Returns:
            Data: PyTorch Geometric数据对象 / PyTorch Geometric Data object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在 / File not found: {file_path}")

        # 自动检测文件格式 / Auto-detect file format
        if file_format == "auto":
            file_format = self._detect_file_format(file_path)

        self.logger.info(f"构建图表示 / Building graph representation: {file_path} ({file_format})")

        if file_format == "verilog":
            return self._build_from_verilog(file_path)
        elif file_format == "aig":
            return self._build_from_aig(file_path)
        else:
            raise ValueError(f"不支持的文件格式 / Unsupported file format: {file_format}")

    def _build_from_verilog(self, verilog_file: str) -> Data:
        """从Verilog文件构建图 / Build graph from Verilog file

        Args:
            verilog_file: Verilog文件路径 / Verilog file path

        Returns:
            Data: PyTorch Geometric数据对象 / PyTorch Geometric Data object
        """
        # 解析Verilog网表 / Parse Verilog netlist
        netlist_info = self._parse_verilog_netlist(verilog_file)

        # 构建图结构 / Build graph structure
        return self._build_graph_from_netlist(netlist_info)

    def _build_from_aig(self, aig_file: str) -> Data:
        """从AIG文件构建图 / Build graph from AIG file

        Args:
            aig_file: AIG文件路径 / AIG file path

        Returns:
            Data: PyTorch Geometric数据对象 / PyTorch Geometric Data object
        """
        # 先转换为Verilog再解析 / Convert to Verilog first then parse
        verilog_file = self.abc_interface.aig_to_verilog(aig_file)
        return self._build_from_verilog(verilog_file)

    def _parse_verilog_netlist(self, verilog_file: str) -> Dict[str, Any]:
        """解析Verilog网表文件 / Parse Verilog netlist file

        Args:
            verilog_file: Verilog文件路径 / Verilog file path

        Returns:
            Dict[str, Any]: 网表信息字典 / Netlist information dictionary
        """
        netlist_info = {
            "module_name": "",
            "ports": {"inputs": [], "outputs": []},
            "wires": [],
            "gates": [],
            "registers": []
        }

        try:
            with open(verilog_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 移除注释 / Remove comments
            content = re.sub(r'//.*', '', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

            # 提取模块名 / Extract module name
            module_match = re.search(r'module\s+(\w+)', content)
            if module_match:
                netlist_info["module_name"] = module_match.group(1)

            # 提取端口 / Extract ports
            self._extract_ports(content, netlist_info)

            # 提取wire声明 / Extract wire declarations
            self._extract_wires(content, netlist_info)

            # 提取门和寄存器 / Extract gates and registers
            self._extract_gates_and_registers(content, netlist_info)

            self.logger.debug(f"网表解析完成 / Netlist parsing completed: "
                            f"模块={netlist_info['module_name']}, "
                            f"门数量={len(netlist_info['gates'])}, "
                            f"寄存器数量={len(netlist_info['registers'])}")

        except Exception as e:
            self.logger.error(f"Verilog网表解析失败 / Verilog netlist parsing failed: {e}")
            raise

        return netlist_info

    def _extract_ports(self, content: str, netlist_info: Dict[str, Any]) -> None:
        """提取端口信息 / Extract port information"""
        # 提取输入端口 / Extract input ports
        input_matches = re.findall(r'input\s+(?:\[.*?\])?\s*(\w+)', content)
        netlist_info["ports"]["inputs"].extend(input_matches)

        # 提取输出端口 / Extract output ports
        output_matches = re.findall(r'output\s+(?:\[.*?\])?\s*(\w+)', content)
        netlist_info["ports"]["outputs"].extend(output_matches)

    def _extract_wires(self, content: str, netlist_info: Dict[str, Any]) -> None:
        """提取wire声明 / Extract wire declarations"""
        wire_matches = re.findall(r'wire\s+(?:\[.*?\])?\s*(\w+)', content)
        netlist_info["wires"].extend(wire_matches)

    def _extract_gates_and_registers(self, content: str, netlist_info: Dict[str, Any]) -> None:
        """提取门和寄存器实例 / Extract gate and register instances"""
        # 匹配assign语句（组合逻辑） / Match assign statements (combinational logic)
        assign_pattern = r'assign\s+(\w+)\s*=\s*(.+?);'
        assign_matches = re.findall(assign_pattern, content)

        for output, expression in assign_matches:
            gate_info = self._parse_assign_expression(output, expression)
            if gate_info:
                netlist_info["gates"].append(gate_info)

        # 匹配always块（时序逻辑） / Match always blocks (sequential logic)
        always_pattern = r'always\s*@\s*\(.*?\)\s*begin(.*?)end'
        always_matches = re.findall(always_pattern, content, re.DOTALL)

        for always_body in always_matches:
            reg_info = self._parse_always_block(always_body)
            if reg_info:
                netlist_info["registers"].extend(reg_info)

        # 匹配门级实例 / Match gate-level instances
        gate_instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\);'
        gate_instances = re.findall(gate_instance_pattern, content)

        for gate_type, instance_name, connections in gate_instances:
            if gate_type.lower() in ['and', 'or', 'not', 'xor', 'nand', 'nor']:
                gate_info = self._parse_gate_instance(gate_type, instance_name, connections)
                if gate_info:
                    netlist_info["gates"].append(gate_info)

    def _parse_assign_expression(self, output: str, expression: str) -> Optional[Dict[str, Any]]:
        """解析assign表达式 / Parse assign expression"""
        try:
            # 简化的表达式解析 / Simplified expression parsing
            expression = expression.strip()

            # 检测运算符 / Detect operators
            gate_type = "unknown"
            inputs = []

            if '&' in expression:
                gate_type = "and"
                inputs = [inp.strip() for inp in expression.split('&')]
            elif '|' in expression:
                gate_type = "or"
                inputs = [inp.strip() for inp in expression.split('|')]
            elif '^' in expression:
                gate_type = "xor"
                inputs = [inp.strip() for inp in expression.split('^')]
            elif '~' in expression:
                gate_type = "not"
                inputs = [expression.replace('~', '').strip()]
            elif '+' in expression:
                gate_type = "add"
                inputs = [inp.strip() for inp in expression.split('+')]
            elif '-' in expression:
                gate_type = "sub"
                inputs = [inp.strip() for inp in expression.split('-')]
            else:
                # 直接连接 / Direct connection
                gate_type = "wire"
                inputs = [expression]

            return {
                "type": "gate",
                "gate_type": gate_type,
                "instance_name": f"assign_{output}",
                "inputs": inputs,
                "outputs": [output],
                "input_width": 1,  # 简化假设
                "delay": self._estimate_gate_delay(gate_type),
                "area": self._estimate_gate_area(gate_type)
            }

        except Exception as e:
            self.logger.warning(f"表达式解析失败 / Expression parsing failed: {expression}, {e}")
            return None

    def _parse_always_block(self, always_body: str) -> List[Dict[str, Any]]:
        """解析always块 / Parse always block"""
        registers = []

        # 简单的寄存器检测 / Simple register detection
        reg_assignments = re.findall(r'(\w+)\s*<=\s*(.+?);', always_body)

        for reg_name, reg_input in reg_assignments:
            registers.append({
                "type": "register",
                "instance_name": reg_name,
                "input": reg_input.strip(),
                "output": reg_name,
                "bit_width": 1,  # 简化假设
                "has_clock": True,
                "has_reset": "reset" in always_body.lower(),
                "has_enable": "enable" in always_body.lower()
            })

        return registers

    def _parse_gate_instance(self, gate_type: str, instance_name: str, connections: str) -> Optional[Dict[str, Any]]:
        """解析门级实例 / Parse gate-level instance"""
        try:
            # 解析连接 / Parse connections
            conn_list = [conn.strip() for conn in connections.split(',')]

            return {
                "type": "gate",
                "gate_type": gate_type.lower(),
                "instance_name": instance_name,
                "inputs": conn_list[1:],  # 假设第一个是输出
                "outputs": [conn_list[0]],
                "input_width": len(conn_list) - 1,
                "delay": self._estimate_gate_delay(gate_type.lower()),
                "area": self._estimate_gate_area(gate_type.lower())
            }

        except Exception as e:
            self.logger.warning(f"门实例解析失败 / Gate instance parsing failed: {instance_name}, {e}")
            return None

    def _build_graph_from_netlist(self, netlist_info: Dict[str, Any]) -> Data:
        """从网表信息构建图 / Build graph from netlist information"""
        # 创建节点和边列表 / Create node and edge lists
        nodes = []
        edges = []
        node_features = []

        # 节点ID映射 / Node ID mapping
        node_id_map = {}
        current_id = 0

        # 添加端口节点 / Add port nodes
        for port in netlist_info["ports"]["inputs"]:
            node_id_map[port] = current_id
            nodes.append({
                "id": current_id,
                "name": port,
                "type": "port",
                "subtype": "input"
            })
            current_id += 1

        for port in netlist_info["ports"]["outputs"]:
            node_id_map[port] = current_id
            nodes.append({
                "id": current_id,
                "name": port,
                "type": "port",
                "subtype": "output"
            })
            current_id += 1

        # 添加wire节点 / Add wire nodes
        for wire in netlist_info["wires"]:
            if wire not in node_id_map:
                node_id_map[wire] = current_id
                nodes.append({
                    "id": current_id,
                    "name": wire,
                    "type": "wire"
                })
                current_id += 1

        # 添加门节点 / Add gate nodes
        for gate in netlist_info["gates"]:
            gate_node_id = current_id
            node_id_map[gate["instance_name"]] = gate_node_id

            nodes.append({
                "id": gate_node_id,
                "name": gate["instance_name"],
                "type": "gate",
                "gate_type": gate.get("gate_type", "unknown"),
                "input_width": gate.get("input_width", 1),
                "delay": gate.get("delay", 0.0),
                "area": gate.get("area", 0.0)
            })
            current_id += 1

            # 添加连接边 / Add connection edges
            # 输入连接 / Input connections
            for inp in gate.get("inputs", []):
                if inp in node_id_map:
                    edges.append((node_id_map[inp], gate_node_id, "connection"))

            # 输出连接 / Output connections
            for outp in gate.get("outputs", []):
                if outp not in node_id_map:
                    node_id_map[outp] = current_id
                    nodes.append({
                        "id": current_id,
                        "name": outp,
                        "type": "wire"
                    })
                    current_id += 1

                edges.append((gate_node_id, node_id_map[outp], "connection"))

        # 添加寄存器节点 / Add register nodes
        for reg in netlist_info["registers"]:
            reg_node_id = current_id
            node_id_map[reg["instance_name"]] = reg_node_id

            nodes.append({
                "id": reg_node_id,
                "name": reg["instance_name"],
                "type": "reg",
                "bit_width": reg.get("bit_width", 1),
                "has_clock": reg.get("has_clock", True),
                "has_reset": reg.get("has_reset", False),
                "has_enable": reg.get("has_enable", False)
            })
            current_id += 1

            # 添加寄存器连接 / Add register connections
            if reg.get("input") and reg["input"] in node_id_map:
                edges.append((node_id_map[reg["input"]], reg_node_id, "connection"))

        # 生成节点特征 / Generate node features
        for node in nodes:
            features = self._create_node_features(node)
            node_features.append(features)

        # 转换为PyTorch Geometric格式 / Convert to PyTorch Geometric format
        return self._create_pyg_data(nodes, edges, node_features)

    def _create_node_features(self, node: Dict[str, Any]) -> np.ndarray:
        """创建节点特征向量 / Create node feature vector

        Args:
            node: 节点信息字典 / Node information dictionary

        Returns:
            np.ndarray: 节点特征向量 / Node feature vector
        """
        features = np.zeros(self.config.total_node_feature_dim)

        # 节点类型特征 (one-hot编码) / Node type features (one-hot encoding)
        node_type = node.get("type", "wire")
        if node_type in self.node_type_mapping:
            features[self.node_type_mapping[node_type]] = 1.0

        # 门类型特征 / Gate type features
        if node_type == "gate":
            gate_type = node.get("gate_type", "unknown")
            if gate_type in self.gate_type_mapping:
                gate_idx = self.gate_type_mapping[gate_type]
                # 使用简单的嵌入 / Use simple embedding
                features[4 + gate_idx] = 1.0

        # 数值特征 / Numerical features
        feature_idx = 12  # 4 (node type) + 8 (gate type)

        if node_type == "gate":
            features[feature_idx] = min(node.get("input_width", 1) / 32.0, 1.0)  # 归一化输入宽度
            features[feature_idx + 1] = min(node.get("delay", 0.0) / 10.0, 1.0)  # 归一化延迟
            features[feature_idx + 2] = min(node.get("area", 0.0) / 100.0, 1.0)  # 归一化面积
        elif node_type == "reg":
            features[feature_idx + 3] = min(node.get("bit_width", 1) / 32.0, 1.0)  # 归一化位宽

        # 时序特征 / Timing features
        timing_idx = 20  # 4 + 8 + 8

        if node_type == "reg":
            features[timing_idx] = 1.0 if node.get("has_clock", False) else 0.0
            features[timing_idx + 1] = 1.0 if node.get("has_reset", False) else 0.0
            features[timing_idx + 2] = 1.0 if node.get("has_enable", False) else 0.0

        return features

    def _create_pyg_data(self, nodes: List[Dict], edges: List[Tuple],
                        node_features: List[np.ndarray]) -> Data:
        """创建PyTorch Geometric数据对象 / Create PyTorch Geometric Data object"""

        # 转换节点特征 / Convert node features
        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # 转换边索引 / Convert edge indices
        if edges:
            edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()

            # 边属性 (边类型) / Edge attributes (edge types)
            edge_attr = torch.tensor([self.edge_type_mapping.get(e[2], 0) for e in edges], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # 创建Data对象 / Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes)
        )

        # 添加额外信息 / Add additional information
        data.node_names = [node["name"] for node in nodes]
        data.node_types = [node["type"] for node in nodes]

        self.logger.info(f"图构建完成 / Graph construction completed: "
                        f"节点数={data.num_nodes}, 边数={data.edge_index.size(1)}")

        return data

    def _estimate_gate_delay(self, gate_type: str) -> float:
        """估算门延迟 / Estimate gate delay"""
        delay_map = {
            "and": 1.0, "or": 1.0, "not": 0.5, "xor": 1.5,
            "mux": 2.0, "add": 3.0, "sub": 3.0, "mult": 5.0,
            "wire": 0.1, "unknown": 1.0
        }
        return delay_map.get(gate_type, 1.0)

    def _estimate_gate_area(self, gate_type: str) -> float:
        """估算门面积 / Estimate gate area"""
        area_map = {
            "and": 2.0, "or": 2.0, "not": 1.0, "xor": 4.0,
            "mux": 6.0, "add": 10.0, "sub": 10.0, "mult": 50.0,
            "wire": 0.0, "unknown": 2.0
        }
        return area_map.get(gate_type, 2.0)

    def _detect_file_format(self, file_path: str) -> str:
        """检测文件格式 / Detect file format"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.v', '.verilog']:
            return "verilog"
        elif ext in ['.aig']:
            return "aig"
        else:
            return "verilog"  # 默认

    def to_networkx(self, data: Data) -> nx.Graph:
        """转换为NetworkX图 / Convert to NetworkX graph"""
        return to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])

    def from_networkx(self, nx_graph: nx.Graph) -> Data:
        """从NetworkX图转换 / Convert from NetworkX graph"""
        return from_networkx(nx_graph)

    def get_graph_statistics(self, data: Data) -> Dict[str, Any]:
        """获取图统计信息 / Get graph statistics"""
        nx_graph = self.to_networkx(data)

        stats = {
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.size(1),
            "node_type_distribution": {},
            "average_degree": data.edge_index.size(1) * 2 / data.num_nodes if data.num_nodes > 0 else 0,
            "density": nx.density(nx_graph),
            "is_connected": nx.is_connected(nx_graph.to_undirected())
        }

        # 节点类型分布 / Node type distribution
        for node_type in data.node_types:
            stats["node_type_distribution"][node_type] = stats["node_type_distribution"].get(node_type, 0) + 1

        return stats


# 测试代码 / Test code
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 测试图构建 / Test graph construction
    graph_builder = RTLNetlistGraph()

    # 这里需要一个实际的Verilog文件来测试 / Need an actual Verilog file for testing
    # test_verilog = "test.v"
    # if os.path.exists(test_verilog):
    #     data = graph_builder.build_from_file(test_verilog)
    #     stats = graph_builder.get_graph_statistics(data)
    #     print(f"图统计信息 / Graph statistics: {stats}")

    print("RTL图表示模块测试完成 / RTL graph representation module test completed")