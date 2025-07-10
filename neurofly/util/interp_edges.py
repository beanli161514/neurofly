import numpy as np
import argparse
from neurofly.dbio import read_nodes, get_edges_by, add_edges, add_nodes, get_max_nid, delete_edges
from neurofly.image_reader import wrap_image
from brightest_path_lib.algorithm import NBAStarSearch
from tqdm import tqdm


def interp_edge(edge,image,interval=3,dis_thres=6):
    '''
    edges: [[x,y,z],[x,y,z]], coordinates of two nodes
    image: wrapped image class

    given edges labeled manually, first calculate its length, then interpolate long edges by sampling from the brightest path between the two coordinates found by NBAStar algorithm.
    return the interpolated coordinates as a list
    '''
    border_size = 5
    [src, tar] = edge
    # calculate image roi of this edge
    xmin = min(src[0],tar[0]) - border_size
    xmax = max(src[0],tar[0]) + border_size
    ymin = min(src[1],tar[1]) - border_size
    ymax = max(src[1],tar[1]) + border_size
    zmin = min(src[2],tar[2]) - border_size
    zmax = max(src[2],tar[2]) + border_size
    roi = [xmin,ymin,zmin,xmax-xmin,ymax-ymin,zmax-zmin]
    img = image.from_roi(roi)
    offset = [xmin,ymin,zmin]
    src_l = [i-j for i,j in zip(src,offset)]
    tar_l = [i-j for i,j in zip(tar,offset)]
    dis = np.sqrt(sum([(i-j)**2 for i,j in zip(src_l,tar_l)]))

    if dis<dis_thres or dis>300:
        return [src,tar]
    
    sa = NBAStarSearch(img,src_l,tar_l)
    path = None
    try:
        path = sa.search()
    except:
        print(f"Can't solve path from {src} to {tar}")
        return [src, tar]

    if path is not None:
        path = np.vstack(path) + np.array(offset)
        path = path.tolist()
        sampled_points = path[:-(interval-1):interval]
        sampled_points.append(path[-1])
    return sampled_points



def interp_all(db_path,image_path,interval=3,dis_thres=6):
    image = wrap_image(image_path)
    # find all edges labeled manually
    nodes = read_nodes(db_path)
    nodes = {n['nid']: n for n in nodes}
    edges = get_edges_by(db_path)
    edges = [[e['src'],e['des']] for e in edges]
    edges = [edge for edge in edges if edge[0]<edge[1]]
    max_id = get_max_nid(db_path)
    for edge in tqdm(edges):
        [src,tar] = [nodes[edge[0]]['coord'],nodes[edge[1]]['coord']]
        interped_points = interp_edge([src,tar],image,interval,dis_thres)
        # get new nids for interpolated nodes, then add new nodes and edges to the db
        if len(interped_points)<=2:
            continue
        node_ids = [edge[0]] + list(range(max_id+1,max_id+len(interped_points)-1)) + [edge[1]]
        max_id += len(interped_points)
        added_nodes = []
        added_edges = []
        for i, node in enumerate(interped_points[1:-1],start=1): 
            added_nodes.append(
                {
                    'nid': node_ids[i],
                    'coord': node,
                    'creator': 'astar',
                    'type': 0,
                    'checked': 0
                }
            )
        
        for n1,n2 in zip(node_ids[:-1],node_ids[1:]):
            added_edges.append([n1,n2])
        
        delete_edges(db_path,[edge])
        add_nodes(db_path,added_nodes)
        add_edges(db_path,added_edges,user_name='astar')



def command_line_interface():
    parser = argparse.ArgumentParser(description="args for interpolation")
    parser.add_argument('-db_path', type=str, default=None, help="path to segs dataset")
    parser.add_argument('-image_path', type=str, default=None, help="path to the input image, only zarr, ims, tif are currently supported")
    parser.add_argument('-interval', type=int, default=3, help="interval of sampled")
    parser.add_argument('-dis_thres', type=int , default=6, help="minimal distance between nodes for intropolation")
    args = parser.parse_args()
    interp_all(args.db_path,args.image_path,args.interval,args.dis_thres)
