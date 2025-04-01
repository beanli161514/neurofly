from datetime import datetime
import networkx as nx
import numpy as np
from tqdm import tqdm

from .sqliteDBIO import sqliteDBIO

class NeuroDB:
    def __init__(self, db_name:str, lazy_load_G:bool=True):
        self.DB = None
        self.switch_to(db_name)

        self._G = None
        if not lazy_load_G:
            self.init_graph()
    
    def switch_to(self, db_name:str):
        if db_name is None:
            return
        if self.DB is not None:
            self.DB.switch_to(db_name)
            self._G = None
        else:
            if db_name.endswith('.db'):
                self.DB = sqliteDBIO(db_name)
            else:
                raise ValueError('Unsupported DB format.')
        
    @property
    def G(self):
        if self._G is None:
            self.init_graph()
        return self._G
    
    def init_graph(self):
        if not self.DB:
            raise ValueError('DB not initialized.')
        if self._G is None:
            self._G = nx.Graph()
        NODES = self.read_nodes()
        EDGES = self.read_edges()
        for node in NODES:
            self._G.add_node(
                node['nid'], 
                coord=node['coord'], 
                creator=node['creator'], 
                type=node['type'], 
                checked=node['checked'], 
                status=node['status'], 
                date=node['date']
            )
        for edge in EDGES:
            self._G.add_edge(edge['src'], edge['dst'], creator=edge['creator'], date=edge['date'])

    def add_nodes(self, nodes:list[dict]):
        # nodes: [{'nid', 'coord', 'creator', 'type', 'checked', 'status', 'date'}]
        date = datetime.now()
        for n in nodes:
            if 'date' not in n:
                n['date'] = date
        self.DB.add_nodes(nodes)
        for node in nodes:
            self.G.add_node(
                node['nid'], 
                coord=node['coord'], 
                creator=node['creator'], 
                type=node['type'], 
                checked=node['checked'], 
                status=node['status'], 
                date=node['date']
            )
    
    def add_edges(self, edges:list[dict]):
        # edges: [{'src', 'dst', 'creator', 'date'}]
        date = datetime.now()
        for e in edges:
            if 'date' not in e:
                e['date'] = date
        self.DB.add_edges(edges)
        for edge in edges:
            self.G.add_edge(edge['src'], edge['dst'], creator=edge['creator'], date=edge['date'])
    
    def delete_nodes(self, nids:list):
        self.DB.delete_nodes(nids)
        self.G.remove_nodes_from(nids)
    
    def delete_edges(self, edges:list):
        self.DB.delete_edges(edges)
        for src, dst in edges:
            self.G.remove_edge(src, dst)
    
    def read_nodes(self, filter:str=None):
        return self.DB.read_nodes(filter)
    
    def read_edges(self, creator:str=None):
        return self.DB.read_edges(creator)

    def update_nodes(self, nids:list, creator:str=None, type:int=None, checked:int=None, status:int=None):
        def _update_nodes_in_graph(key:str, value:any=None, date:datetime=None):
            if value and (value != self.G.nodes[nid][key]):
                self.G.nodes[nid][key] = value
                self.G.nodes[nid]['date'] = date
        date = datetime.now()
        self.DB.update_nodes(nids, creator, type, checked, status, date)
        for nid in nids:
            _update_nodes_in_graph('creator', creator, date)
            _update_nodes_in_graph('type', type, date)
            _update_nodes_in_graph('checked', checked, date)
            _update_nodes_in_graph('status', status, date)
    
    def check_node(self, nid:int):
        date = datetime.now()
        self.DB.check_node(nid, date)
        self.G.nodes[nid]['checked'] = 1
        self.G.nodes[nid]['date'] = date
    
    def uncheck_nodes(self, nids:list[int]):
        date = datetime.now()
        self.DB.uncheck_nodes(nids, date)
        for nid in nids:
            self.G.nodes[nid]['checked'] = -1
            self.G.nodes[nid]['date'] = date

    def segs2db(self, segs):
        self.DB.segs2db(segs)

    def connect_segs(self):
        def _cal_angle(v1:np.ndarray, v2:np.ndarray):
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            v1 = v1 / norm_v1
            v2 = v2 / norm_v2
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
            return angle

        angle_disconnt = 100
        angle_connt_valid = 45
        angle_connt_invalid = 120
        dist_threshold = 12

        # check all path nodes
        path_nids = [node for node, degree in self.G.degree() if degree==2]
        nodes_invalid = []
        pbar = tqdm(path_nids, desc='proofreading')
        for nid in pbar:
            if self.G.degree[nid]!=2:
                continue
            nbr_nids = list(self.G.neighbors(nid))
            v1 = np.asarray(self.G.nodes[nbr_nids[0]]['coord']) - np.asarray(self.G.nodes[nid]['coord'])
            v2 = np.asarray(self.G.nodes[nbr_nids[1]]['coord']) - np.asarray(self.G.nodes[nid]['coord'])
            angle = _cal_angle(v1, v2)
            # if invalid
            if angle < angle_disconnt:
                self.delete_edges([[nid, nbr_nids[0]], [nid, nbr_nids[1]]])
                nodes_invalid.append([nid, angle])
            pbar.set_description(f'proofreading, num of invalid edges: {len(nodes_invalid)*2}')
        
        # try to connect end nodes
        edges_autoConnt = []
        end_nids = [node for node, degree in self.G.degree() if degree==1]
        pbar = tqdm(end_nids)
        for end_nid in pbar:
            if self.G.degree[end_nid]>1:
                continue
            curr_path_nids = [end_nid] + [des for src, des in list(nx.dfs_edges(self.G, end_nid, depth_limit=5))]
            curr_coords = np.asarray([self.G.nodes[nid]['coord'] for nid in curr_path_nids])
            offset = [i-dist_threshold//2 for i in curr_coords[0]]
            roi = offset + [i+dist_threshold for i in offset]
            curr_direction = np.sum(curr_coords[:-1:3] - curr_coords[1::3], axis=0)

            nbr_nid_list = set(self.DB.read_nid_within_roi(roi)) - set(curr_path_nids)
            nbr_nid_list = [nid for nid in nbr_nid_list if self.G.degree[nid]==1]
            matched_nbr_nid = None
            min_angle = angle_connt_valid
            for nbr_nid in nbr_nid_list:
                nbr_path_nids = [nbr_nid] + [des for src, des in list(nx.dfs_edges(self.G, nbr_nid, depth_limit=5))]
                nbr_coords = np.asarray([self.G.nodes[nid]['coord'] for nid in nbr_path_nids])
                nbr_direction = np.sum(nbr_coords[1::3] - nbr_coords[:-1:3], axis=0)

                direction_angle = _cal_angle(curr_direction, nbr_direction)
                connection_angle = min(
                    _cal_angle(v1=curr_coords[1]-curr_coords[0], v2=nbr_coords[0]-curr_coords[0]),
                    _cal_angle(v1=nbr_coords[1]-nbr_coords[0], v2=curr_coords[0]-nbr_coords[0]),
                )
                # if valid
                if direction_angle<=min_angle and connection_angle>=angle_connt_invalid:
                    min_angle = direction_angle
                    matched_nbr_nid = nbr_nid

            if matched_nbr_nid is not None:
                self.add_edges([{'src':end_nid, 'dst':matched_nbr_nid, 'creator':'connector'}])
                self.uncheck_nodes([end_nid, matched_nbr_nid])
                edges_autoConnt.append([end_nid, matched_nbr_nid])
            pbar.set_description(f'auto connecting, num of auto connected segs: {len(edges_autoConnt)}')

        print(f'Remove {len(nodes_invalid)*2} invalid edges.\nAuto Connect {len(edges_autoConnt)} segments.')
        return nodes_invalid, edges_autoConnt

    def get_annotation_info(self, len_threshold:int=0):
        connected_components = list(nx.connected_components(self.G))
        valid_cc = []
        for cc in connected_components:
            if len(cc) < len_threshold:
                continue
            valid = True
            for nid in cc:
                if (self.G.degree(nid)==1 and self.G.nodes[nid]['checked']==0) or (self.G.nodes[nid]['checked']==-1):
                    valid = False
                    break
            if valid:
                valid_cc.append(cc)
        info = []
        for cc in valid_cc:
            sub_G:nx.Graph = self.G.subgraph(cc)
            length = 0
            for src, des in sub_G.edges:
                length += np.linalg.norm(np.array(sub_G.nodes[src]['coord']) - np.array(sub_G.nodes[des]['coord']))
            branch_nid = [nid for nid in sub_G.nodes if sub_G.degree(nid) > 2]
            end_nid = [nid for nid in sub_G.nodes if sub_G.degree(nid) == 1]
            info.append({
                'nid': list(cc),
                'branch_nid': branch_nid,
                'end_nid': end_nid,
                'length': int(length)
            })
        info.sort(key=lambda x:x['length'], reverse=True)
        return info


