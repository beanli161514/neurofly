from datetime import datetime
import sqlite3
import os
import glob
import numpy as np


def segs2db(segs,path):
    '''
    given a list of segs, add all nodes and edges to the datebase.
    seg:
        {
            points: [head,...,tail],
            sampled_points: points[::interval],
        }
    node:
        {
            nid: int, PRIMARY KEY
            coord: str,
            creator: str,
            status: int, # 1 for show, 0 for hidden(removed)
            type: int, # 1 for Soma, 0 for normal node
            date: str, TIMESTAMP
            checked: int
        }
    edge:
        {
            src: int, 
            des: int,
            date: str, TIMESTAMP
            creator: str,
            PRIMARY KEY: (src,des)
        }

    the graph is undirected, thus edges exist in pairs
    '''

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS segs(
                sid INTEGER PRIMARY KEY,
                points TEXT,
                sampled_points TEXT
            )
            '''
        )
    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS nodes(
                nid INTEGER PRIMARY KEY,
                coord TEXT,
                creator TEXT,
                status INTEGER,
                type INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checked INTEGER
            )
            '''
        )

    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS edges(
                src INTEGER,
                des INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                creator TEXT,
                PRIMARY KEY (src,des)
            )
            '''
        )

    query = f"SELECT COUNT(*) FROM segs"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]
    
    for seg in segs:
        count+=1
        cursor.execute(f"INSERT INTO segs (sid, points, sampled_points) VALUES (?, ?, ?)",
                    (count, sqlite3.Binary(str(seg['points']).encode()), sqlite3.Binary(str(seg['sampled_points']).encode())))

    print(f'Number of segs in database: {count}, {len(segs)} newly added.')

    query = f"SELECT COUNT(*) FROM nodes"
    cursor.execute(query)
    result = cursor.fetchone()
    count = result[0]

    conn.commit()
    conn.close()

    # assign unique nid for each node in segs according to index
    nodes = [] # [nid,coord,nbr_ids]
    edges = [] # [source_id,target_id]

    for seg in segs:
        points = seg['sampled_points']
        if len(points)>=2:
            count+=1
            nodes.append([count,points[0]])
            edges.append([count,count+1])

            for c in points[1:-1]:
                count+=1
                nodes.append([count,c])
                edges.append([count,count+1])

            count+=1
            nodes.append([count,points[-1]])
        else:
            count+=1
            nodes.append([count,points[0]])
    

    # add nodes and edges to the database
    nodes_list = []
    for node in nodes:
        nodes_list.append({
            'nid': node[0],
            'coord': node[1],
            'creator': 'seger',
            'status': 1,
            'type': 0,
            'checked': 0
        })

    print(f'Adding {len(nodes)} nodes to database')
    add_nodes(path,nodes_list)
    print(f'Adding {len(edges)} edges to database')
    add_edges(path,edges,user_name='seger')


def read_segs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM segs ORDER BY sid")
    rows = cursor.fetchall()
    segs = []
    for row in rows:
        data = {
            'sid': row[0],
            'points': eval(row[1]),
            'sampled_points': eval(row[2]),
        }
        segs.append(data)
    conn.close()
    return segs



def read_nodes(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes ORDER BY nid")
    rows = cursor.fetchall()
    points = []
    for row in rows:
        data = {
            'nid': row[0],
            'coord': eval(row[1]),
            'creator': row[2],
            'status': row[3],
            'type': row[4],
            'date': row[5],
            'checked': row[6]
        }
        points.append(data)
    conn.close()
    return points


def read_edges(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM edges")
    rows = cursor.fetchall()
    edges = []
    for row in rows:
        data = {
            'src': row[0],
            'des': row[1],
            'date': row[2],
            'creator': row[3],
        }
        edges.append(data)
    conn.close()
    return edges



def delete_nodes(path,node_ids):
    # given a list of node_ids, delete nodes from nodes table and edges from edges table
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM nodes WHERE nid IN ({})".format(','.join(map(str, node_ids))))
    # Remove edges where either source or destination node is in the given list
    cursor.execute("DELETE FROM edges WHERE src IN ({}) OR des IN ({})".format(','.join(map(str, node_ids)), ','.join(map(str, node_ids))))
    conn.commit()
    conn.close()



def delete_edges(path, edges):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    for edge in edges:
        src, des = edge
        # Delete both (src, des) and (des, src) if they exist
        cursor.execute("DELETE FROM edges WHERE src=? AND des=?", (src, des))
        cursor.execute("DELETE FROM edges WHERE src=? AND des=?", (des, src))
    conn.commit()
    conn.close()


'''
CREATE TABLE IF NOT EXISTS nodes(
    nid INTEGER PRIMARY KEY,
    coord TEXT,
    creator TEXT,
    status INTEGER,
    type INTEGER,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checked INTEGER
)
'''


def add_nodes(path,nodes):
    # given a list of nodes, write them to node table
    # nodes: [{'nid','coord','creator','status','type'}]
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    time = datetime.now()
    for node in nodes:
        if type(node['nid']) != int:
            print(f"{node} is illegal")
            continue
        cursor.execute(f"INSERT INTO nodes (nid, coord, creator, status, type, date, checked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    node['nid'],
                    sqlite3.Binary(str(node['coord']).encode()),
                    node['creator'],
                    node['status'] if 'status' in node.keys() else 1,
                    node['type'],
                    time,
                    node['checked'] if 'checked' in node.keys() else 0
                )
            )
    conn.commit()
    conn.close()


def add_edges(path, edges, user_name='somebody'):
    # given list of edges, write them to edges table
    # edges: [[src,tar]]
    undirected_edges = []
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    time = datetime.now()

    for [src, tar] in edges:
        undirected_edges.append([src, tar])
        undirected_edges.append([tar, src])

    # for [src, tar] in edges:
    #     if isinstance(src, int) and isinstance(tar, int):
    #         cursor.execute("SELECT COUNT(*) FROM nodes WHERE nid = ?", (src,))
    #         src_exists = cursor.fetchone()[0]

    #         cursor.execute("SELECT COUNT(*) FROM nodes WHERE nid = ?", (tar,))
    #         tar_exists = cursor.fetchone()[0]

    #         if src_exists and tar_exists:
    #             # Add undirected edges if both src and tar exist
    #             undirected_edges.append([src, tar])
    #             undirected_edges.append([tar, src])
    #         else:
    #             print(f"Either {src} or {tar} does not exist in the nodes table.")
    #     else:
    #         print(f"{[src, tar]} is illegal")

    # Insert the valid undirected edges into the edges table
    for edge in undirected_edges:
        cursor.execute(
            "INSERT OR IGNORE INTO edges (src, des, date, creator) VALUES (?, ?, ?, ?)",
            (edge[0], edge[1], time, user_name)
        )

    conn.commit()
    conn.close()


def get_max_nid(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Retrieve the highest existing nid value
    cursor.execute("SELECT MAX(nid) FROM nodes")
    max_nid = cursor.fetchone()[0] or 0  # If there are no existing items, set max_nid to 0
    conn.commit()
    conn.close()
    return max_nid



def check_node(path,nid):
    # given list of edges, write them to edges table
    # edges: [[src,tar]]
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE OR IGNORE nodes SET checked = 1 WHERE nid = ?", (nid,))

    conn.commit()
    conn.close()


def uncheck_nodes(path,nids):
    # given list of node ids, label them as unchecked
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE OR IGNORE nodes SET checked = -1 WHERE nid IN ({})".format(','.join(map(str, nids))))

    conn.commit()
    conn.close()


def change_type(path,nid,type):
    # given node id, change node type
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE nodes SET type = ? WHERE nid = ?", (type, nid))

    conn.commit()
    conn.close()


def change_creator(path,nid,creator):
    # given node id, change node type
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE nodes SET creator = ? WHERE nid = ?", (creator, nid))

    conn.commit()
    conn.close()


def change_status(path,nid,status):
    # given node id, change node status
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("UPDATE nodes SET status = ? WHERE nid = ?", (status, nid))

    conn.commit()
    conn.close()


def get_edges_by(db_path, creator=None):
    '''
    if creator is left empty, get all edges labeled manually
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if creator is None:
        cursor.execute("SELECT * FROM edges WHERE creator != ?", ('seger',))
    else:
        cursor.execute("SELECT * FROM edges WHERE creator=?", (creator,))
    rows = cursor.fetchall()
    edges = []
    for row in rows:
        data = {
            'src': row[0],
            'des': row[1],
            'date': row[2],
            'creator': row[3],
        }
        edges.append(data)
    conn.close()
    return edges



def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS nodes(
                nid INTEGER PRIMARY KEY,
                coord TEXT,
                creator TEXT,
                status INTEGER,
                type INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checked INTEGER
            )
            '''
        )

    cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS edges(
                src INTEGER,
                des INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                creator TEXT,
                PRIMARY KEY (src,des)
            )
            '''
        )
    conn.close()


def swc2db(swc_dir,db_path,creator='unknown'):
    '''
    swc files to database file, facilitating data manipulation
    swc: nid type x y z radius parent
    type: 1 for soma, 2 for axon, 3 for (basal) dendrite, 4 for apical dendrite
    swc file format: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    in database: 1 for Soma 0 for others
    '''
    initialize_db(db_path)
    swc_files = glob.glob(os.path.join(swc_dir,'*.swc'), recursive=True)
    for swc_file in swc_files:
        data = np.loadtxt(swc_file)[:,:7]
        # check structure, each file should have exactly one soma
        num_of_soma = np.sum(data[:,1] == 1)
        if num_of_soma != 1:
            print(f'This .swc file contains {num_of_soma} somas, which is insane')
            continue
        nodes = []
        edges = []
        if os.path.isfile(db_path):
            nid_offset = get_max_nid(db_path) + 2
        else:
            nid_offset = 0

        for [nid,type,x,y,z,radius,parent] in data.tolist():
            # nodes: [{'nid','coord','type','checked'},...]
            # edge: [[src,des],...], username
            nodes.append({
                'nid': int(nid) + nid_offset,
                'coord': [x,y,z],
                'creator': creator,
                'status': 1,
                'type': 1 if type == 1 else 0,
                'checked': 1
            })
            edges.append([nid+nid_offset,parent+nid_offset])
            edges.append([parent+nid_offset,nid+nid_offset])

        add_nodes(db_path,nodes)
        add_edges(db_path,edges,'terafly')

