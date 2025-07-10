import numpy as np
import random
import networkx as nx
from neurofly.dbio import read_nodes,read_edges


def draw_frame(roi,viewer,width=1,color='blue',scale=[1,1,1]):
    # roi: [x_offset, y_offset, z_offset, x_size, y_size, z_size]
    x_offset, y_offset, z_offset, x_size, y_size, z_size = roi
    # Calculate the coordinates of the vertices of the cuboid
    vertices = np.array([
        [x_offset, y_offset, z_offset],                          
        [x_offset + x_size, y_offset, z_offset],                 
        [x_offset + x_size, y_offset + y_size, z_offset],        
        [x_offset, y_offset + y_size, z_offset],                 
        [x_offset, y_offset, z_offset + z_size],                 
        [x_offset + x_size, y_offset, z_offset + z_size],        
        [x_offset + x_size, y_offset + y_size, z_offset + z_size], 
        [x_offset, y_offset + y_size, z_offset + z_size]         
    ])

    edges = np.array([[0,1,2,3,0],[1,2,6,5,1],[5,6,7,4,5],[4,7,3,0,4],[0,1,5,4,0],[2,6,7,3,2]])

    viewer.add_shapes(
        data=vertices[edges],
        shape_type='path',
        edge_color=color,
        edge_width=width,
        face_color='transparent',
        opacity=1,
        scale=scale
    )


def show_segs_as_instances(segs,viewer,size=0.8):
    '''
    segs: [
        [[x,y,z],[x,y,z],...],
        ...
    ]
    '''
    points = []
    colors = []
    num_segs = 0
    num_branches = 0
    for seg in segs:
        seg_color = random.random()
        points+=seg
        colors+=[seg_color for _ in seg]
        if len(seg)>=2:
            num_segs+=1
        if len(seg)==1:
            num_branches+=1

    colors = np.array(colors)
    colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    properties = {
        'colors': colors
    }

    print(f'num of segs (length >= 2): {num_segs}')
    print(f'num of branch points: {num_branches}')
    print(f'num of points: {len(points)}')
    point_layer = viewer.add_points(np.array(points),ndim=3,face_color='colors',size=size,border_color='colors',shading='spherical',border_width=0,properties=properties,face_colormap='turbo')


def show_segs_as_paths(segs,viewer,width=1):
    '''
    segs: [
        [[x,y,z],[x,y,z],...],
        ...
    ]
    '''
    paths = []
    colors = []
    num_segs = 0
    num_branches = 0
    length = 0
    for seg in segs:
        seg_color = random.random()
        if len(seg)>=2:
            num_segs+=1
            paths.append(np.array(seg))
            colors.append(seg_color)
            length+=len(seg)*3
        if len(seg)==1:
            num_branches+=1
        length+=9

    colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    properties = {
        'colors': colors
    }

    path_layer = viewer.add_shapes(
        paths, properties=properties, shape_type='path', edge_width=width, edge_color='colors', edge_colormap='turbo', blending='opaque'
    )
    print(f'num of segs (length >= 2): {num_segs}')
    print(f'num of branch points: {num_branches}')
    print(f'num of points: {length}')



def show_graph_as_paths(neurites,viewer,len_thres=10):
    segs = []
    seg_colors = []
    nodes = []
    node_colors = []
    G = neurites.G
    connected_components = list(nx.connected_components(G))

    for cc in connected_components:
        # extract segs and branch nodes, assign same color
        if len(cc)<=len_thres:
            continue
        sub_g = G.subgraph(cc).copy()
        color = random.random()
        spanning_tree = nx.minimum_spanning_tree(sub_g, algorithm='kruskal', weight=None)
        # remove circles by keeping only DFS tree
        sub_g.remove_edges_from(set(sub_g.edges) - set(spanning_tree.edges))
        branch_nodes = [node for node, degree in sub_g.degree() if degree >= 3]
        nodes += [G.nodes[i]['coord'] for i in branch_nodes]
        node_colors += [color]*len(branch_nodes)
        sub_g.remove_nodes_from(branch_nodes)

        cc = list(nx.connected_components(sub_g))
        for ns in cc:
            sub_sub_g = sub_g.subgraph(ns)
            end_nodes = [node for node, degree in sub_sub_g.degree() if degree == 1]
            if (len(end_nodes)!=2):
                continue
            path = nx.shortest_path(sub_sub_g, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
            seg_points = [G.nodes[i]['coord'] for i in path]
            # add branch points back
            source_nbrs = list(G.neighbors(end_nodes[0]))
            branch_node = list(set(source_nbrs)-set(path))
            if len(branch_node)==1:
                seg_points.insert(0,G.nodes[branch_node[0]]['coord'])

            target_nbrs = list(G.neighbors(end_nodes[1]))
            branch_node = list(set(target_nbrs)-set(path))
            if len(branch_node)==1:
                seg_points.append(G.nodes[branch_node[0]]['coord'])

            seg_colors.append(color)
            segs.append(seg_points)


    seg_colors = (seg_colors-np.min(seg_colors))/(np.max(seg_colors)-np.min(seg_colors))
    properties = {
        'colors': seg_colors
    }

    path_layer = viewer.add_shapes(
        segs, properties=properties, shape_type='path', edge_width=1, edge_color='colors', edge_colormap='turbo', blending='opaque'
    )



def vis_edges_by_creator(viewer,db_path,color_dict):
    '''
    visualize edges by their creators
    color_dict: {
        'creator1': color1,
        'creator2': color2,
        'default': default_color
        ...
    }
    '''
    # find all edges labeled manually
    nodes = read_nodes(db_path)
    edges = read_edges(db_path)
    nodes = {n['nid']: n for n in nodes}
    edges = [[e['src'],e['des'],e['creator']] for e in edges]
    edges = [edge for edge in edges if edge[0]<edge[1]]

    edge_length = {key:0 for key,_ in color_dict.items()}

    vectors = []
    v_colors = []
    for edge in edges:
        [src,tar,creator] = [nodes[edge[0]]['coord'],nodes[edge[1]]['coord'],edge[2]]
        v = [j-i for i,j in zip(src,tar)]
        p = src
        vectors.append([p,v])
        if creator in color_dict.keys():
            v_colors.append(color_dict[creator])
            edge_length[creator]+=1
        else:
            v_colors.append(color_dict['default'])
            edge_length[creator]+=1 
    viewer.add_vectors(vectors,edge_color=v_colors,edge_width=2,vector_style='line')
