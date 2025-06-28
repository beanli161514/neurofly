import os
import sqlite3
from datetime import datetime

class NeurodbSQLite:
    def __init__(self, db_path):
        if db_path is not None:
            self.switch_to(db_path)
        else:
            return
        
        db_exists = os.path.exists(db_path)
        if not db_exists:
            self.init_db()

    def switch_to(self, db_path):
        self.db_path = db_path

    def init_db(self):
        """Initialize the database with tables and indexes"""
        self.init_table()
        self.init_index()
        self.init_spatial_index()
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
    
    def init_action_table(self):
        """Initialize the action table for tracking changes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS actions(
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            )
            '''
        )
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
                    'points': sqlite3.Binary(str(seg_data['points']).encode()),
                    'version': seg_data.get('version', 1),
                    'date': seg_data.get('date', date)
                })
            cursor.executemany(
                "INSERT INTO segs (sid, points, version, date) " \
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
                x, y, z = node_data['x'], node_data['y'], node_data['z']
                entries.append({
                    'nid': nid,
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'creator': node_data['creator'],
                    'type': node_data['type'],
                    'checked': node_data['checked'],
                    'status': node_data['status'],
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
        except Exception as e:
            print(f"Error in add_nodes: {e}")
            conn.rollback()
            conn.close()
            raise e

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
        except Exception as edge_data:
            print(f"Error in add_edges: {edge_data}")
            conn.rollback()
            conn.close()
            raise edge_data

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
                SELECT n.nid, n.x, n.y, n.z
                FROM nodes n JOIN nodes_rtree r ON n.nid = r.id
                WHERE r.minX >= ? AND r.minX <= ?
                AND r.minY >= ? AND r.minY <= ?
                AND r.minZ >= ? AND r.minZ <= ?
            '''
        else:
            # Fallback to standard range query
            query = '''
                SELECT nid, x, y, z FROM nodes
                WHERE x BETWEEN ? AND ?
                AND y BETWEEN ? AND ?
                AND z BETWEEN ? AND ?
            '''
        cursor.execute(query, (x_min, x_max, y_min, y_max, z_min, z_max))
        nodes = {}
        for row in cursor.fetchall():
            nodes[row['nid']] = {
                'coord': [row['x'], row['y'], row['z']],
            }

        nids = list(nodes.keys())
        query = f"SELECT src, dst FROM edges WHERE src IN ({','.join(map(str, nids))}) OR dst IN ({','.join(map(str, nids))})"
        cursor.execute(query)
        edges = {}
        for row in cursor.fetchall():
            edges[(row['src'], row['dst'])] = {}
            
        conn.close()
        return nodes, edges
    
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

    def delete_edges(self, edges):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            for src, dst in edges:
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
            params = []
            if creator is not None:
                update_parts.append("creator = ?")
                where_conditions.append("(creator IS NULL OR creator != ?)")
                params.extend([creator, creator])
            if type is not None:
                update_parts.append("type = ?")
                where_conditions.append("(type IS NULL OR type != ?)")
                params.extend([type, type])
            if checked is not None:
                update_parts.append("checked = ?")
                where_conditions.append("(checked IS NULL OR checked != ?)")
                params.extend([checked, checked])
            if status is not None:
                update_parts.append("status = ?")
                where_conditions.append("(status IS NULL OR status != ?)")
                params.extend([status, status])
            if cid is not None:
                update_parts.append("cid = ?")
                where_conditions.append("(cid IS NULL OR cid != ?)")
                params.extend([cid, cid])
            if date is None:
                date = datetime.now()
            update_parts.append("date = ?")
            params.append(date)
            
            placeholders = ','.join('?' for _ in nids)
            if where_conditions:
                additional_conditions = f" AND ({' OR '.join(where_conditions)})"
                query = f"UPDATE nodes SET {', '.join(update_parts)} WHERE nid IN ({placeholders}){additional_conditions}"
            else:
                query = f"UPDATE nodes SET {', '.join(update_parts)} WHERE nid IN ({placeholders})"
            params.extend(nids)
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error in update_nodes: {e}")
            conn.rollback()
            conn.close()
            raise e
    
    def check_node(self, nid:int, date:datetime=None):
        self.update_nodes([nid], checked=1, date=date)
    
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
                'points': sqlite3.Binary(str(seg['points']).encode()),
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
    
    def read_tasks(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = """
            WITH relevant_cid AS (
                SELECT DISTINCT cid
                FROM nodes
                WHERE checked = -1
            ),
            cid_cnt AS (
                SELECT cid, COUNT(*) AS cnnt_len
                FROM nodes
                WHERE cid IN (SELECT cid FROM relevant_cid)
                GROUP BY cid
            )
            SELECT n.nid, n.x, n.y, n.z, c.cnnt_len
            FROM nodes n
            JOIN cid_cnt c ON n.cid = c.cid
            WHERE n.checked = -1
            ORDER BY c.cnnt_len DESC;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        tasks = []
        for row in rows:
            tasks.append({
                'nid': row['nid'],
                'coord': [row['x'], row['y'], row['z']],
                'cnnt_len': row['cnnt_len']
            })
        conn.close()
        return tasks
    
if __name__ == "__main__":
    db_path = '/home/ryuuyou/Project/neurofly_dev/test/seg_dev_test/out/test.db'
    neurodb = NeurodbSQLite(db_path)
    tasks = neurodb.read_tasks()
    print(f"Number of tasks: {len(tasks)}")
    for task in tasks[0:10]:
        print(task)

    