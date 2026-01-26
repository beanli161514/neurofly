# for SEED Project

import os
from tqdm import tqdm
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from rtree import index as rtree_index
from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.neurodb.image_reader import Ims

def shift_SEGS(SEGS:dict, offset:int):
    SEGS_shift = {}
    sid_shift_map = {}
    for sid, attr in SEGS.items():
        sid_shift = sid + offset
        sid_shift_map[sid] = sid_shift
        attr['sid'] = sid_shift
        SEGS_shift[sid_shift] = attr
    return SEGS_shift, sid_shift_map

def shift_NODES(NODES:dict, offset:int, sid_shift_map:dict):
    NODES_shift = {}
    nid_shift_map = {}
    for nid, attr in NODES.items():
        nid_shift = nid + offset
        nid_shift_map[nid] = nid_shift
        attr['nid'] = nid_shift
        sid = attr['sid']
        if sid in sid_shift_map and sid > 0:
            sid_shift = sid_shift_map[sid]
            attr['sid'] = sid_shift
            attr['cid'] = sid_shift
        NODES_shift[nid_shift] = attr
    return NODES_shift, nid_shift_map

def shift_EDGES(EDGES:dict, nid_shift_map:dict):
    EDGES_shift = {}
    for (src_nid, dst_nid), attr in EDGES.items():
        src_nid_shift = nid_shift_map[src_nid]
        dst_nid_shift = nid_shift_map[dst_nid]
        attr['src'] = src_nid_shift
        attr['dst'] = dst_nid_shift
        EDGES_shift[(src_nid_shift, dst_nid_shift)] = attr
    return EDGES_shift

def DBSliced_to_DBMerged(DB_Merged:NeurodbSQLite, DB_Sliced:NeurodbSQLite, ROI:list=None):
    max_sid, _ = DB_Merged.get_max_sid_version()
    SEGS = DB_Sliced.read_segs(sids='*')
    SEGS_shift, sid_shift_map = shift_SEGS(SEGS, offset=max_sid)

    max_nid = DB_Merged.get_max_nid()
    if ROI is None:
        NODES = DB_Sliced.read_nodes(nids='*')
        EDGES = DB_Sliced.read_edges_by_nids(nids=list(NODES.keys()))
    else:
        NODES, EDGES = DB_Sliced.read_nodes_edges_within_roi(roi=ROI)
    NODES_shift, nid_shift_map = shift_NODES(NODES, offset=max_nid, sid_shift_map=sid_shift_map)
    EDGE_shift = shift_EDGES(EDGES, nid_shift_map=nid_shift_map)

    DB_Merged.add_segs(SEGS_shift)
    DB_Merged.add_nodes(NODES_shift)
    DB_Merged.add_edges(EDGE_shift)

def Merge_Main(img_path:str, db_merged_path:str, db_sliced_dir:str):
    IMG = Ims(img_path)
    level = 0
    IMG_SHAPE = IMG.info[level]['image_size']
    X,Y,Z = IMG_SHAPE
    SLICE_THICKNESS = 300
    Z_COORDS = list(np.arange(Z//SLICE_THICKNESS) * SLICE_THICKNESS)
    ROIs = []
    for _z in Z_COORDS:
        _roi = [0,0,int(_z)] + [X,Y,SLICE_THICKNESS]
        ROIs.append(_roi)

    DB_Merged = NeurodbSQLite(db_merged_path)
    db_sliced_name = sorted(os.listdir(db_sliced_dir))
    print(db_sliced_name)
    for name, roi in tqdm(zip(db_sliced_name, ROIs)):
        db_sliced_path = os.path.join(db_sliced_dir, name)
        DB_Sliced = NeurodbSQLite(db_sliced_path)
        DBSliced_to_DBMerged(DB_Merged, DB_Sliced, roi)

def cnnt_landmark(img_path:str, db_merged_path:str):
    IMG = Ims(img_path)
    DB = NeurodbSQLite(db_merged_path)
    def __get_G_with_RTree__(_NODES:dict, _EDGES:dict):
        __NeuroG = NeuroGraph(_NODES, _EDGES)
        __G = __NeuroG.graph
        __p = rtree_index.Property(dimension=3)
        __rtree_data = [(__attr['nid'],tuple(__attr['coord']+__attr['coord']),None) for __nid, __attr in _NODES.items()]
        __RTree = rtree_index.Index(__rtree_data, properties=__p)
        return __G, __RTree
    
    def __match__(_pre_nids, _post_nids, _G:nx.Graph):
        DIST_THRESHOLD = 100
        __pre_coords = np.array([_G.nodes[nid]['coord'] for nid in _pre_nids])
        __post_coords = np.array([_G.nodes[nid]['coord'] for nid in _post_nids])
        __distances = cdist(__pre_coords, __post_coords, metric='euclidean')
        __idx_pre2post = np.argmin(__distances, axis=1)
        __dist_pre2post = np.min(__distances, axis=1)
        __EDGES = {}
        for __pre_idx, (__post_idx, __dist) in enumerate(zip(__idx_pre2post, __dist_pre2post)):
            if __dist > DIST_THRESHOLD:
                continue
            __pre_nid = _pre_nids[__pre_idx]
            __post_nid = _post_nids[__post_idx]
            if __pre_nid > __post_nid:
                __pre_nid, __post_nid = __post_nid, __pre_nid
            __EDGES[(__pre_nid, __post_nid)] = {'creator':'admin', 'dist':__dist}

        __idx_post2pre = np.argmin(__distances, axis=0)
        __dist_post2pre = np.min(__distances, axis=0)
        for __post_idx, (__pre_idx, __dist) in enumerate(zip(__idx_post2pre, __dist_post2pre)):
            if __dist > DIST_THRESHOLD:
                continue
            __post_nid = _post_nids[__post_idx]
            __pre_nid = _pre_nids[__pre_idx]
            if __pre_nid > __post_nid:
                __pre_nid, __post_nid = __post_nid, __pre_nid
            if (__pre_nid, __post_nid) not in __EDGES:
                __EDGES[(__pre_nid, __post_nid)] = {'creator':'admin', 'dist':float(__dist)}
        return __EDGES
    
    level = 0
    IMG_SHAPE = IMG.info[level]['image_size']
    X,Y,Z = IMG_SHAPE
    SLICE_THICKNESS = 300
    Z_COORDS = (np.arange(1, Z//SLICE_THICKNESS) * SLICE_THICKNESS).tolist()
    Z_RANGE = 10
    ROIs = []
    for _z in Z_COORDS:
        # [offet_x, offet_y, offset_z] + [size_x, size_y, size_z]
        _fuse_roi = [0,0,_z-SLICE_THICKNESS] + [X,Y,2*SLICE_THICKNESS]
        # [start_x, start_y, start_z] + [end_x, end_y, end_z]
        _pre_merge_roi = [0,0,_z-Z_RANGE] + [X,Y,_z]
        # [start_x, start_y, start_z] + [end_x, end_y, end_z]
        _post_merge_roi = [0,0,_z] + [X,Y,_z+Z_RANGE]
        ROIs.append([_fuse_roi, _pre_merge_roi, _post_merge_roi])
    TASK_NIDS = []
    for _fuse_roi, _pre_merge_roi, _post_merge_roi in tqdm(ROIs):
        _NODES, _EDGES = DB.read_nodes_edges_within_roi(_fuse_roi)
        _G, _RTree = __get_G_with_RTree__(_NODES, _EDGES)
        _pre_end_nids = [_nid for _nid in list(_RTree.intersection(_pre_merge_roi, objects=False)) if _G.degree[_nid]==1]
        _post_end_nids = [_nid for _nid in list(_RTree.intersection(_post_merge_roi, objects=False)) if _G.degree[_nid]==1]
        _EDGES_matched = __match__(_pre_end_nids, _post_end_nids, _G)
        # print(f'=== {len(_EDGES_matched)} ===')
        for (_src, _dst), _attr in _EDGES_matched.items():
            TASK_NIDS.append(_src)
            TASK_NIDS.append(_dst)
            # print(f'{_attr['dist']:.4f}')
        DB.add_edges(_EDGES_matched)
    TASK_NIDS = list(dict.fromkeys(TASK_NIDS))
    return TASK_NIDS

def reset_tasks(db_merged_path:str, append_task_nids:list=[]):
    DB_Merged = NeurodbSQLite(db_merged_path)
    SOMA_NODES = DB_Merged.read_nodes(ntype=1)
    CC_TASKS_NIDS = []
    for soma_nid in tqdm(SOMA_NODES.keys(), desc='loading CC with soma'):
        CC_nodes, CC_edges = DB_Merged.read_connected_components(soma_nid, with_edges=True)
        CC_G = NeuroGraph(CC_nodes, CC_edges)
        tasks = []
        # branch nodes in cycle
        cycles = list(nx.simple_cycles(CC_G.graph))
        for _cycle in cycles:
            for _nid in _cycle:
                if CC_G.graph.degree(_nid) > 2:
                    tasks.append(_nid)
        if bool(cycles):
            print(f'{len(cycles)} cycles detected in CC with soma({soma_nid})')
            print(f'{len(tasks)} branch nodes extracted from the cycles')
        # endnodes
        nid_deg_1 = [_nid for _nid in CC_G.graph.nodes if CC_G.graph.degree(_nid)==1]
        tasks += nid_deg_1
        nid_deg_3 = [_nid for _nid in CC_G.graph.nodes if CC_G.graph.degree(_nid)>2]
        tasks += nid_deg_3
        CC_TASKS_NIDS.append(tasks)
    CC_TASKS_NIDS = [__nid for _nids in sorted(CC_TASKS_NIDS, reverse=True) for __nid in _nids]
    TASKS_NIDS = list(dict.fromkeys(CC_TASKS_NIDS + append_task_nids))
    TASKS_NODES = DB_Merged.read_nodes(nids=TASKS_NIDS)
    TASKS = {}
    for nid, attr in tqdm(TASKS_NODES.items(), desc='formating tasks'):
        TASKS[nid] = {
            'nid': nid,
            'coord': attr['coord'],
            'sid': attr['sid'],
            'checked': -1,
            'creator': 'admin',
        }
    DB_Merged.add_tasks(TASKS, 'w')
    print('down')
