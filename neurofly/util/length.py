from tqdm import tqdm
import numpy as np
import networkx as nx

from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.util.interpolation import FSM
from neurofly.util.data_conversion import parse_swc
    
def cal_angle_degree(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    if np.linalg.norm(ba)==0 or np.linalg.norm(bc)==0:
        return 0.0
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def cal_length_from_swc_noInterp(SWC:list):
    NODES = parse_swc(SWC)
    length_total = 0.0
    for nid, node in NODES.items():
        n_coord = np.asarray(node['coord'])
        pid = node['pid']
        if pid is not None:
            p_coord = np.asarray(NODES[pid]['coord'])
            length = np.linalg.norm(n_coord - p_coord)
            length_total += length
    return length_total

def cal_length_from_graph(G:nx.Graph):
    if isinstance(G, NeuroGraph):
        G = G.graph
    G_degree = dict(G.degree())
    deg2_nodes = {_nid for _nid, _deg in G_degree.items() if _deg == 2}
    subG_deg2 = G.subgraph(deg2_nodes)

    length_total = 0
    for CC in tqdm(nx.connected_components(subG_deg2), total=nx.number_connected_components(subG_deg2), desc="Calculating length"):
        end_points = []
        for src in CC:
            for dst in G.neighbors(src):
                if dst not in CC:
                    end_points.append(dst)
        # if there are not exactly two end points, skip this component
        if len(end_points) != 2:
            continue

        src, dst = end_points
        path_nids = nx.shortest_path(G, src, dst)
        path_coords = np.asarray([G.nodes[_nid]['coord'] for _nid in path_nids])
        interp_degree = 4 if len(path_coords) > 4 else 2
        sample_num = len(path_coords)*6
        path_coords_interp, _, _ = FSM(path_coords, degree=interp_degree, sample_num=sample_num)
        for src_coord, dst_coord in zip(path_coords_interp[:-1], path_coords_interp[1:]):
            length = np.linalg.norm(dst_coord - src_coord)
            length_total += length
    
    return length_total

def cal_length_from_swc_interp(SWC:list, *, return_log:bool=False):
    NODES = parse_swc(SWC)
    G = nx.Graph()
    ANGLE_THRESHOLD = 90
    logger = []
    for nid, node in tqdm(NODES.items(), total=len(NODES), desc="Removing invalid edges for length calculation"):
        n_coord = np.asarray(node['coord'])
        if not G.has_node(nid):
            G.add_node(nid, coord=n_coord, type=node['type'])
        pid = node['pid']
        if pid is None:
            continue
        else:
            p_coord = np.asarray(NODES[pid]['coord'])
            if not G.has_node(pid):
                G.add_node(pid, coord=p_coord, type=NODES[pid]['type'])
            sid = node['sid']
            ppid = NODES[pid]['pid']
            if sid is None or ppid is None:
                G.add_edge(nid, pid)
            else:
                s_coord = np.asarray(NODES[sid]['coord'])
                pp_coord = np.asarray(NODES[ppid]['coord'])
                angle_s_n_p = cal_angle_degree(s_coord, n_coord, p_coord)
                angle_n_p_pp = cal_angle_degree(n_coord, p_coord, pp_coord)
                average_angle = (angle_s_n_p + angle_n_p_pp) / 2
                if average_angle > ANGLE_THRESHOLD:
                    G.add_edge(nid, pid)
                else:
                    logger.append(f'Abnormal edge: {nid} {n_coord.astype(int)} -> {pid} {p_coord.astype(int)}, angle: {average_angle:.2f} degrees, skipped.')

    length_total = cal_length_from_graph(G)

    if return_log:
        return length_total, logger
    else:
        return length_total
