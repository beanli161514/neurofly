import sqlite3
import json
from datetime import datetime

class NeurodbSQLite:
    def __init__(self, db_path):
        if db_path is not None:
            self.switch_to(db_path)
        else:
            return
        
        # db_exists = os.path.exists(db_path)
        # if not db_exists:
        self.init_db()

    def switch_to(self, db_path):
        self.db_path = db_path

    def init_db(self):
        """Initialize the database with tables and indexes"""
        self.init_table()
        self.init_index()
        self.init_spatial_index()
        self.init_action_table_with_index()
        print("Database initialized successfully.")

    def init_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # segs table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS segs(
                sid INTEGER PRIMARY KEY,
                points TEXT,
                version INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        # nodes table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS nodes(
                nid INTEGER PRIMARY KEY,
                x INTEGER,
                y INTEGER,
                z INTEGER,
                creator TEXT,
                type INTEGER,
                checked INTEGER,
                status INTEGER,
                sid INTEGER DEFAULT NULL,
                cid INTEGER DEFAULT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sid) REFERENCES segs(sid)
            )
            '''
        )
        # edges table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS edges(
                src INTEGER,
                dst INTEGER,
                creator TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (src,dst),
                FOREIGN KEY (src) REFERENCES nodes(nid),
                FOREIGN KEY (dst) REFERENCES nodes(nid),
                CHECK (src <= dst)
            )
            '''
        )
        conn.commit()
        conn.close()
    
    def init_action_table_with_index(self):
        """Initialize the action table for tracking changes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS actions(
                aid INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                task_nid INTEGER,
                task_node TEXT,
                action_nid INTEGER DEFAULT NULL,
                action_node TEXT,
                action_edge_src INTEGER DEFAULT NULL,
                action_edge_dst INTEGER DEFAULT NULL,
                path_nodes TEXT,
                path_edges TEXT,
                creator TEXT DEFAULT 'tester',
                history TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_nid) REFERENCES nodes(nid),
                FOREIGN KEY (action_nid) REFERENCES nodes(nid),
                FOREIGN KEY (action_edge_src) REFERENCES nodes(nid),
                FOREIGN KEY (action_edge_dst) REFERENCES nodes(nid)
            )
            '''
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_task_nid ON actions (task_nid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_action_nid ON actions (action_nid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_type ON actions (type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_creator ON actions (creator)")
        conn.commit()
        conn.close()
    
    def init_index(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_nid ON nodes (nid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_checked ON nodes (checked)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_sid ON nodes (sid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_cid ON nodes (cid)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges (src)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges (dst)")

        conn.commit()
        conn.close()

    def init_spatial_index(self):
        """Add spatial indexing to an existing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if R-tree module is available
        cursor.execute('PRAGMA module_list')
        modules = cursor.fetchall()
        rtree_available = any('rtree' in module[0].lower() for module in modules)
        
        if not rtree_available:
            print("SQLite R-tree module not available. Using standard index instead.")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_xyz ON nodes (x, y, z)")
            conn.commit()
            conn.close()
            return False
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_rtree'")
        if cursor.fetchone() is not None:
            print("R-tree table already exists")
            conn.close()
            return True
        
        try:
            # Rtree spatial index
            cursor.execute(
                '''
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_rtree USING rtree(
                    id,              -- Integer primary key
                    minX, maxX,      -- X coordinate bounds
                    minY, maxY,      -- Y coordinate bounds
                    minZ, maxZ       -- Z coordinate bounds
                )
                '''
            )
            # insert trigger
            cursor.execute(
                '''
                CREATE TRIGGER IF NOT EXISTS nodes_rtree_insert AFTER INSERT ON nodes
                BEGIN
                    INSERT INTO nodes_rtree VALUES (
                        new.nid,
                        new.x, new.x,
                        new.y, new.y, 
                        new.z, new.z
                    );
                END;
                '''
            )
            # delete trigger
            cursor.execute(
                '''
                CREATE TRIGGER IF NOT EXISTS nodes_rtree_delete AFTER DELETE ON nodes
                BEGIN
                    DELETE FROM nodes_rtree WHERE id = old.nid;
                END;
                '''
            )
            # Populate the R-tree with existing nodes
            print("Populating R-tree index with existing nodes...")
            cursor.execute(
                '''
                INSERT OR IGNORE INTO nodes_rtree
                SELECT nid, x, x, y, y, z, z FROM nodes
                '''
            )
            conn.commit()
            print("Spatial index created successfully.")
            return True
        except Exception as e:
            print(f"Error creating spatial index: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def add_segs(self, segs:dict):
        # given a list of segs, write them to segs table
        # segs: {sid: {'sid', 'points', 'version', 'date'}}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            date = datetime.now()
            entries = []
            for sid, seg_data in segs.items():
                seg_data: dict
                entries.append({
                    'sid': sid,
                    'points': json.dumps(seg_data['points']),
                    'version': seg_data.get('version', 1),
                    'date': seg_data.get('date', date)
                })
            cursor.executemany(
                "INSERT INTO segs (sid, points, version, date) " + 
                "VALUES (:sid, :points, :version, :date)",
                entries
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error in add_segs: {e}")
            conn.rollback()
            conn.close()
            raise e

    def add_nodes(self, nodes:dict):
        # given a list of nodes, write them to node table
        # nodes: {nid: {'nid', 'x', 'y', 'z', 'creator', 'type', 'checked', 'status', 'sid', 'cid', 'date'}}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            date = datetime.now()
            entries = []
            for nid, node_data in nodes.items():
                node_data: dict
                if 'coord' in node_data:
                    x, y, z = node_data['coord']
                elif 'x' in node_data and 'y' in node_data and 'z' in node_data: 
                    x, y, z = node_data['x'], node_data['y'], node_data['z']
                entries.append({
                    'nid': nid,
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'creator': node_data['creator'],
                    'type': node_data['type'],
                    'checked': node_data['checked'],
                    'status': node_data.get('status', 1),
                    'sid': node_data.get('sid', None),
                    'cid': node_data.get('cid', None),
                    'date': node_data.get('date', date)
                })
            cursor.executemany(
                "INSERT INTO nodes (nid, x, y, z, creator, type, checked, status, sid, cid, date) " +
                "VALUES (:nid, :x, :y, :z, :creator, :type, :checked, :status, :sid, :cid, :date)",
                entries
            )
            conn.commit()
            conn.close()
        except Exception as E:
            print(f"Error in add_nodes: {E}")
            conn.rollback()
            conn.close()
            raise E

    def add_edges(self, edges:dict):
        # given list of edges, write them to edges table
        # edges: {(src, dst): {'src', 'dst', 'creator', 'date'}}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            date = datetime.now()
            entries = []
            for (src, dst), edge_data in edges.items():
                edge_data: dict
                if src > dst:
                    src, dst = dst, src
                entries.append({
                    'src': src,
                    'dst': dst,
                    'creator': edge_data['creator'],
                    'date': edge_data.get('date', date)
                })
            cursor.executemany(
                "INSERT INTO edges (src, dst, creator, date) VALUES (:src, :dst, :creator, :date)",
                entries
            )
            conn.commit()
            conn.close()
        except Exception as E:
            print(f"Error in add_edges: {E}")
            conn.rollback()
            conn.close()
            raise E
    
    def add_actions(self, actions:list):
        def __dictWithTupleKey2list__(_dict:dict):
            _list = [[k,v] for k,v in _dict.items()]
            return _list
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            date = datetime.now()
            entries = []
            for action in actions:
                action: object
                entries.append({
                    'type': action.action_type,
                    'task_nid': action.task_node['nid'],
                    'task_node': json.dumps(action.task_node),
                    'action_nid': action.action_node['nid'] if action.action_node else None,
                    'action_node': json.dumps(action.action_node) if action.action_node else None,
                    'action_edge_src': action.action_edge['src'] if action.action_edge else None,
                    'action_edge_dst': action.action_edge['dst'] if action.action_edge else None,
                    'path_nodes': json.dumps(action.path_nodes) if action.path_nodes else None,
                    'path_edges': json.dumps(__dictWithTupleKey2list__(action.path_edges)) if action.path_edges else None,
                    'creator': action.creator,
                    'history': json.dumps(
                            {'nodes':action.history['nodes'],
                             'edges':__dictWithTupleKey2list__(action.history['edges'])}
                        ) if action.history else None,
                    'date': action.date if action.date else date
                })
            cursor.executemany(
                "INSERT INTO actions " +
                    "(type, task_nid, task_node, action_nid, action_node, action_edge_src, action_edge_dst, " +
                        "path_nodes, path_edges, creator, history, date) " +
                "VALUES (:type, :task_nid, :task_node, :action_nid, :action_node, :action_edge_src, :action_edge_dst, " +
                        ":path_nodes, :path_edges, :creator, :history, :date)",
                entries
            )
            conn.commit()
            conn.close()
        except Exception as E:
            print(f"Error in add_actions: {E}")
            conn.rollback()
            conn.close()
            raise E
    
    def read_nodes(self, nids:list):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = f"SELECT * FROM nodes WHERE nid IN ({','.join('?' for _ in nids)})"
        cursor.execute(query, nids)
        rows = cursor.fetchall()
        nodes = {}
        for row in rows:
            nid = row['nid']
            nodes[nid] = {
                'nid': nid,
                'coord': [row['x'], row['y'], row['z']],
                'creator': row['creator'],
                'type': row['type'],
                'checked': row['checked'],
                'status': row['status'],
                'sid': row['sid'],
                'cid': row['cid'],
                'date': row['date']
            }
        conn.close()
        return nodes

    def read_one_node(self, nid:int):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE nid=?", (nid,))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            return None
        node = {
            'nid': row['nid'],
            'coord': [row['x'], row['y'], row['z']],
            'creator': row['creator'],
            'type': row['type'],
            'checked': row['checked'],
            'status': row['status'],
            'sid': row['sid'],
            'cid': row['cid'],
            'date': row['date']
        }
        conn.close()
        return node
    
    def read_edges_by_nids(self, nids:list):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        placeholders = ','.join('?' for _ in nids)
        query = f"SELECT * FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})"
        cursor.execute(query, nids + nids)
        
        edges = {}
        for row in cursor.fetchall():
            src, dst = row['src'], row['dst']
            edges[(src, dst)] = {
                'creator': row['creator'],
                'date': row['date']
            }
        conn.close()
        return edges

    def read_one_edge(self, src:int, dst:int=None):
        if dst is None:
            dst = src
            query = "SELECT * FROM edges WHERE src=? OR dst=?"
        else:
            if src > dst:
                src, dst = dst, src
            query = "SELECT * FROM edges WHERE src=? AND dst=?"
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, (src, dst))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            return None
        edge = {
            'src': row['src'],
            'dst': row['dst'],
            'creator': row['creator'],
            'date': row['date']
        }
        conn.close()
        return edge

    def read_nodes_edges_within_roi(self, roi, rtree:bool=True):
        offset, size = roi[:3], roi[-3:]
        x_min, x_max = offset[0], offset[0]+size[0]-1
        y_min, y_max = offset[1], offset[1]+size[1]-1
        z_min, z_max = offset[2], offset[2]+size[2]-1
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_rtree'")
        rtree_exists = cursor.fetchone() is not None
        if rtree and rtree_exists:
            query = '''
                SELECT n.nid, n.x, n.y, n.z, n.type, n.checked
                FROM nodes n JOIN nodes_rtree r ON n.nid = r.id
                WHERE r.minX >= ? AND r.minX <= ?
                AND r.minY >= ? AND r.minY <= ?
                AND r.minZ >= ? AND r.minZ <= ?
            '''
        else:
            # Fallback to standard range query
            query = '''
                SELECT nid, x, y, z, type, checked FROM nodes
                WHERE x BETWEEN ? AND ?
                AND y BETWEEN ? AND ?
                AND z BETWEEN ? AND ?
            '''
        cursor.execute(query, (x_min, x_max, y_min, y_max, z_min, z_max))
        nodes = {}
        for row in cursor.fetchall():
            nodes[row['nid']] = {
                'nid': row['nid'],
                'coord': [row['x'], row['y'], row['z']],
                'type': row['type'],
                'checked': row['checked']
            }

        nids = list(nodes.keys())
        query = f"SELECT src, dst FROM edges WHERE src IN ({','.join(map(str, nids))}) OR dst IN ({','.join(map(str, nids))})"
        cursor.execute(query)
        edges = {}
        for row in cursor.fetchall():
            edges[(row['src'], row['dst'])] = {}
            
        conn.close()
        return nodes, edges
    
    def read_connected_components(self, nid: int, with_edges: bool = False):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        visited_nid = set()
        bfs_queue = [nid]
        nodes = {}
        edges = {}

        while bfs_queue:
            current_nid = bfs_queue.pop(0)
            if current_nid in visited_nid:
                continue
            visited_nid.add(current_nid)
            cursor.execute("SELECT nid, x, y, z, type FROM nodes WHERE nid=?;", (current_nid,))
            row = cursor.fetchone()
            if row:
                nodes[row['nid']] = {
                    'nid': row['nid'],
                    'coord': [row['x'], row['y'], row['z']],
                    'type': row['type']
                }
            cursor.execute(
                """
                SELECT src, dst, creator, date
                FROM edges
                WHERE src=? OR dst=?;
                """,(current_nid, current_nid)
            )
            for edge_row in cursor.fetchall():
                src, dst = edge_row['src'], edge_row['dst']
                edge_key = (src, dst)
                if with_edges and edge_key not in edges:
                    edges[edge_key] = {
                        'creator': edge_row['creator'],
                        'date': edge_row['date']
                    }
                neighbor = dst if src == current_nid else src
                if neighbor not in visited_nid and neighbor not in bfs_queue:
                    bfs_queue.append(neighbor)
        conn.close()

        if with_edges:
            return nodes, edges
        else:
            return nodes

    def read_unchecked_nodes_in_cc(self, nid:int):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = """
            WITH RECURSIVE connected_nodes(nid) AS (
                SELECT ? AS nid
                UNION
                SELECT CASE 
                    WHEN e.src = cn.nid THEN e.dst
                    ELSE e.src
                END AS nid
                FROM connected_nodes cn
                JOIN edges e ON (e.src = cn.nid OR e.dst = cn.nid)
            )
            SELECT cn.nid, n.x, n.y, n.z
            FROM connected_nodes cn
            JOIN nodes n ON cn.nid = n.nid
            WHERE n.checked = -1;
        """
        cursor.execute(query, (nid,))
        nodes = {}
        for row in cursor.fetchall():
            nid = row['nid']
            nodes[nid] = {
                'nid': nid,
                'coord': [row['x'], row['y'], row['z']],
            }
        conn.close()
        return nodes
    
    def delete_nodes(self, nids):
        # given a list of nid, delete nodes from nodes table and edges from edges table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(f"DELETE FROM nodes WHERE nid IN ({','.join(map(str, nids))})")
            # Remove edges where either source or destination node is in the given list
            cursor.execute(f"DELETE FROM edges WHERE src IN ({','.join(map(str, nids))}) OR dst IN ({','.join(map(str, nids))})")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error in delete_nodes: {e}")
            conn.rollback()
            conn.close()
            raise e

    def delete_edges(self, src_dst):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            for src, dst in src_dst:
                if src > dst:
                    src, dst = dst, src
                cursor.execute("DELETE FROM edges WHERE src=? AND dst=?", (src, dst))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error in delete_edges: {e}")
            conn.rollback()
            conn.close()
    
    def update_nodes(self, nids:list, creator:str=None, type:int=None, checked:int=None, status:int=None, cid:int=None, date:datetime=None):
        if all(param is None for param in [creator, type, checked, status, cid]) or not nids:
            return
        if not isinstance(nids, list):
            nids = [nids]
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            update_parts = []
            where_conditions = []
            update_params = []
            condition_params = []
            
            if creator is not None:
                update_parts.append("creator = ?")
                where_conditions.append("(creator IS NULL OR creator != ?)")
                update_params.append(creator)
                condition_params.append(creator)
            if type is not None:
                update_parts.append("type = ?")
                where_conditions.append("(type IS NULL OR type != ?)")
                update_params.append(type)
                condition_params.append(type)
            if checked is not None:
                update_parts.append("checked = ?")
                where_conditions.append("(checked IS NULL OR checked != ?)")
                update_params.append(checked)
                condition_params.append(checked)
            if status is not None:
                update_parts.append("status = ?")
                where_conditions.append("(status IS NULL OR status != ?)")
                update_params.append(status)
                condition_params.append(status)
            if cid is not None:
                update_parts.append("cid = ?")
                where_conditions.append("(cid IS NULL OR cid != ?)")
                update_params.append(cid)
                condition_params.append(cid)
            if date is None:
                date = datetime.now()
            update_parts.append("date = ?")
            update_params.append(date)
            
            placeholders = ','.join('?' for _ in nids)
            if where_conditions:
                additional_conditions = f" AND ({' OR '.join(where_conditions)})"
                query = f"UPDATE nodes SET {', '.join(update_parts)} WHERE nid IN ({placeholders}){additional_conditions}"
                # Correct parameter order: UPDATE params + nids + condition params
                params = update_params + nids + condition_params
            else:
                query = f"UPDATE nodes SET {', '.join(update_parts)} WHERE nid IN ({placeholders})"
                params = update_params + nids

            cursor.execute(query, params)
            # rows_affected = cursor.rowcount
            # print(f"Rows affected: {rows_affected}")
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error in update_nodes: {e}")
            conn.rollback()
            conn.close()
            raise e
    
    def check_nodes(self, nids:list[int], date:datetime=None):
        self.update_nodes(nids, checked=1, date=date)
    
    def uncheck_nodes(self, nids:list[int], date:datetime=None):
        self.update_nodes(nids, checked=-1, date=date)

    def get_max_nid(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Retrieve the highest existing nid value
        cursor.execute("SELECT MAX(nid) FROM nodes")
        max_nid = cursor.fetchone()[0] or 0  # If there are no existing items, set max_nid to 0
        conn.commit()
        conn.close()
        return max_nid
    
    def get_max_sid_version(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(sid) FROM segs")
        max_sid = cursor.fetchone()[0] or 0
        cursor.execute("SELECT MAX(version) FROM segs")
        max_version = cursor.fetchone()[0] or 0
        conn.close()
        return max_sid, max_version
    
    def segs2db(self, segs, version:int=None):
        '''
            segs: [{'points', 'nodes', 'checked', 'edges'}]
        '''
        # insert segs into database
        date = datetime.now()

        # insert segs into database
        max_Sid, max_version = self.get_max_sid_version()
        max_version = version if version is not None else max_version+1
        segs_entries = {}

        max_Nid = self.get_max_nid()
        nodes_entries = {}
        edges_entries = {}

        for seg in segs:
            # segs
            max_Sid += 1
            segs_entries[max_Sid] = {
                'sid': max_Sid,
                'points': seg['points'],
                'version': max_version,
                'date': date
            }
            # nodes
            idx2nid_map = {}
            for cidx, (coord, checked) in enumerate(zip(seg['nodes'], seg['checked'])):
                max_Nid += 1
                x,y,z = coord
                nodes_entries[max_Nid] = {
                    'nid': max_Nid,
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'creator': 'seger',
                    'type': 0,
                    'checked': checked,
                    'status': 1,
                    'sid': max_Sid,
                    'cid': max_Sid,
                    'date': date
                }
                idx2nid_map[cidx] = max_Nid
            # edges
            for src_idx, dst_idx in seg['edges']:
                src_nid = idx2nid_map[src_idx]
                dst_nid = idx2nid_map[dst_idx]
                if src_nid > dst_nid:
                    src_nid, dst_nid = dst_nid, src_nid
                edges_entries[(src_nid, dst_nid)] = {
                    'src': src_nid,
                    'dst': dst_nid,
                    'creator': 'seger',
                    'date': date
                }

        self.add_segs(segs_entries)
        print(f'Number of segs in database: {max_Sid}; {len(segs_entries)} newly added.')
        self.add_nodes(nodes_entries)
        print(f'Adding {len(nodes_entries)} nodes to database')
        self.add_edges(edges_entries)
        print(f'Adding {len(edges_entries)} edges to database')
    
    def read_tasks(self, query:str):
        if not query or query.strip() == '':
            # query = "SELECT nid, x, y, z FROM nodes WHERE checked=-1"
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        tasks = []
        for i, row in enumerate(rows):
            tasks.append({
                'nid': row['nid'],
                'coord': [row['x'], row['y'], row['z']],
            })
        conn.close()
        return tasks
    