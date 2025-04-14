import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from neurofly.dbio import read_edges, read_nodes, uncheck_nodes
from neurofly.image_reader import wrap_image
from rtree import index
from tqdm import tqdm
from brightest_path_lib.algorithm import AStarSearch



class Neurites():
    '''
    Neurites class represents neurites with nodes and associated links between them. It integrates RTree and KDTree for efficient spatial querying and NetworkX graph for exploring and retrieving based on graph structure. This allows both proximity-based searches and graph-based operations.
    '''
    def __init__(self,db_path,image_path=None,require_rtree=True):
        self.db_path = db_path
        if image_path != None:
            self.image = wrap_image(image_path)
        else:
            self.image = None

        nodes = read_nodes(db_path)
        edges = read_edges(db_path)
        
        self.G = nx.Graph() # networkx graph
        for node in nodes:
            self.G.add_node(node['nid'], nid = node['nid'], coord = node['coord'], type = node['type'], checked = node['checked'])

        for edge in edges:
            self.G.add_edge(edge['src'],edge['des'],creator = edge['creator'])

        coord_ids = []
        print("loading nodes")
        p = index.Property(dimension=3)
        rtree_data = []
        for node in nodes:
            coord_ids.append(node['nid'])
            self.G.add_node(node['nid'], nid = node['nid'], coord = node['coord'], type = node['type'], checked = node['checked'], creator = node['creator'])
            rtree_data.append((node['nid'], tuple(node['coord']+node['coord']),None))
        self.coord_ids = coord_ids
        self.rtree = index.Index(rtree_data, properties=p)

    
    def get_skeleton(self,roi=None):
        # get all segments 
        # then interp the sparse points to get dense skeleton
        if roi == None:
            roi = self.image.roi
        segs, _ = self.get_segs_within(roi)
        img = self.image.from_roi(roi)
        skel = []
        for seg in tqdm(segs):
            for src, tar in zip(seg[:-1],seg[1:]):
                sa = AStarSearch(img,src,tar)
                path = None
                try:
                    path = sa.search()
                except:
                    print(f"Can't solve path from {src} to {tar}")
                    path = [src, tar]
                skel += path
        skel = np.array(skel)
        mask = np.zeros_like(img,dtype=np.uint16)
        mask[skel[:, 0], skel[:, 1], skel[:, 2]] = 1
        return mask


    def get_segs_within(self,roi,len_thres=0):
        # get segs within roi, return a list of lists of nodes
        # 1. query nodes within roi
        # 2. generate subgraph
        # 3. remove branches, then traverse every connected components
        # 4. filter out short paths

        nbrs = list(self.rtree.intersection(tuple(roi[0:3]+[i+j for i,j in zip(roi[:3],roi[3:])]), objects=False))
        sub_g = self.G.subgraph(nbrs).copy()
        branch_nodes = [node for node, degree in sub_g.degree() if degree >= 3]
        sub_g.remove_nodes_from(branch_nodes) 
        connected_components = list(nx.connected_components(sub_g))
        segs = []
        for nodes in connected_components:
            end_nodes = [node for node in nodes if sub_g.degree[node] == 1]
            if (len(end_nodes)!=2):
                continue
            path = nx.shortest_path(sub_g, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
            path = [[i-j for i,j in zip(self.G.nodes[node]['coord'],roi[0:3])] for node in path]
            if len(path)>=len_thres:
                segs.append(path)
        # get intensity value
        intens = []
        if self.image is not None:
            img = self.image.from_roi(roi)
            for seg in segs:
                seg_intens = []
                for coord in seg:
                    seg_intens.append(img[coord[0],coord[1],coord[2]])
                intens.append(seg_intens)

        return segs, intens


    def get_segs_by(self,creator,len_thres=0):
        edges_to_include = [(u, v) for u, v, attr in self.G.edges(data=True) if attr.get('creator') == creator]
        sub_g = self.G.edge_subgraph(edges_to_include).copy()
        branch_points = [node for node, degree in sub_g.degree() if degree >= 3]
        sub_g.remove_nodes_from(branch_points)
        connected_components = list(nx.connected_components(sub_g))
        segs = []
        for nodes in connected_components:
            if len(nodes)<=len_thres:
                continue
            subgraph = sub_g.subgraph(nodes).copy()
            end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
            if (len(end_nodes)!=2):
                continue
            path = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
            # path to segment
            segs.append([subgraph.nodes[i]['coord'] for i in path])
        return segs
    

    def uncheck_junctions(self):
        terminals = [node for node, degree in self.G.degree() if degree == 1 and self.G.nodes[node]['checked'] == 0]
        h_size = 6
        unlabeled_junctions = []
        for t in terminals:
            c_coord = self.G.nodes[t]['coord']
            query_box = (c_coord[0]-h_size,c_coord[1]-h_size,c_coord[2]-h_size,c_coord[0]+h_size,c_coord[1]+h_size,c_coord[2]+h_size)
            nbrs = list(self.rtree.intersection(query_box, objects=False))
            cc = list(nx.node_connected_component(self.G, t))
            if len(cc)<=10:
                continue
            nbrs = [i for i in nbrs if i not in cc]
            if len(nbrs)>0:
                nbr_coords = np.array([self.G.nodes[nid]['coord'] for nid in nbrs])
                distances = np.linalg.norm(nbr_coords - np.array(c_coord), axis=1)
                closest_indice = nbrs[np.argsort(distances)[0]]
                if self.G.degree[closest_indice]==2:
                    unlabeled_junctions.append(closest_indice)
        uncheck_nodes(self.db_path, unlabeled_junctions)
        print(f"{len(unlabeled_junctions)} nodes unchecked.")

