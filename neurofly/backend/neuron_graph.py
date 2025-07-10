import networkx as nx
import numpy as np

class NeuroGraph():
    def __init__(self, nodes:dict=None, edges:dict=None):
        self.graph = nx.Graph()
        self.nodes = self.graph.nodes
        self.edges = self.graph.edges

        self.init_graph(nodes, edges)

    def init_graph(self, nodes:dict, edges:dict):
        self.graph.clear()
        if nodes is not None:
            self.add_nodes(nodes)
        if edges is not None:
            self.add_edges(edges)

    def add_nodes(self, nodes:dict):
        for nid, data in nodes.items():
            self.graph.add_node(nid, coord=data['coord'])
    
    def add_edges(self, edges:dict):
        for (src, dst), data in edges.items():
            if src in self.graph and dst in self.graph:
                self.graph.add_edge(src, dst)
    
    def delete_nodes(self, nodes:dict):
        nids = list(nodes.keys())
        for nid in nids:
            if nid in self.graph:
                self.graph.remove_node(nid)
    
    def delete_edges(self, edges:dict):
        src_dst_pairs = list(edges.keys())
        for src, dst in src_dst_pairs:
            if self.graph.has_edge(src, dst):
                self.graph.remove_edge(src, dst)

    def get_connected_components(self, task_node:dict=None):
        cc_nids = list(nx.connected_components(self.graph))
        nodes_nids = []
        nodes_coords = []
        nodes_colors = []
        nodes_sizes = []
        edges_coords = []
        edges_src_nids = []
        edges_dst_nids = []
        for index, _nids in enumerate(cc_nids):
            nodes_nids += _nids
            color = index/len(cc_nids)
            nodes_colors += [color] * len(_nids)
            for _nid in _nids:
                nodes_coords.append(self.graph.nodes[_nid]['coord'])
                if task_node is not None and _nid == task_node['nid']:
                    nodes_sizes.append(2)
                else:
                    nodes_sizes.append(1)
        for src, dst in self.graph.edges:
            src_coord = np.asarray(self.graph.nodes[src]['coord'])
            dst_coord = np.asarray(self.graph.nodes[dst]['coord'])
            edges_coords.append([src_coord, dst_coord-src_coord])
            edges_src_nids.append(src)
            edges_dst_nids.append(dst)

        nodes_coords = np.asarray(nodes_coords)
        edges_coords = np.asarray(edges_coords)

        nodes_properties = {
            'nids': np.asarray(nodes_nids),
            'colors': np.asarray(nodes_colors),
            'sizes': np.asarray(nodes_sizes)
        }

        edges_properties = {
            'src': np.asarray(edges_src_nids),
            'dst': np.asarray(edges_dst_nids)
        }

        return nodes_coords, nodes_properties, edges_coords, edges_properties