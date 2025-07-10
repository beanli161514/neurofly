from .action import Action
from .neuron_graph import NeuroGraph
from ..neurodb.neurodb_sqlite import NeurodbSQLite

class TaskManager():
    def __init__(self, db_path:str=None):
        self.db_path = db_path
        self.DB = NeurodbSQLite(db_path)
        self.G = NeuroGraph()

        self.task_node = {}
        self.action_node = {}
        self.action_edge = {}

        self.action_stack:list[Action] = []

        self.init_task()
    
    def init_graph(self, roi):
        nodes_coords, edges_coords = self.DB.read_nodes_edges_within_roi(roi)
        self.G.init_graph(nodes_coords, edges_coords)

    def init_db_status(self):
        self.MAX_NID = self.DB.get_max_nid()
        self.MAX_SID, _ = self.DB.get_max_sid_version()
    
    def init_task(self):
        self.tasks = self.DB.read_tasks() if self.DB else []
        self.tasks_status = {}
        for idx, task in enumerate(self.tasks):
            self.tasks_status[task['nid']] = {
                'checked': -1,
                'idx': idx
            }
        self.curr_idx = 0
    
    def get_one_task(self, idx:int=None):
        if idx is None:
            idx = self.curr_idx
        task = self.tasks[idx]
        nid = task['nid']
        coord = task['coord']
        seg_len = task['cnnt_len']
        checked = self.tasks_status[nid]['checked']

        self.set_task_node(nid)
        return nid, coord, seg_len, checked

    def get_next_task(self):
        checked = 1
        while checked != -1:
            if self.curr_idx >= len(self.tasks):
                return None, None, None, None
            nid, coord, seg_len, checked = self.get_one_task(self.curr_idx)
            self.set_task_node(nid)
            self.curr_idx += 1
        return nid, coord, seg_len

    def finish_one_task(self, nid:int):
        self.tasks_status[nid]['checked'] = 1

    def set_task_node(self, nid:int):
        """Set the task node for the current task."""
        self.task_node = self.DB.read_one_node(nid)
        return self.task_node

    def set_action_node(self, nid:int):
        """Set the action node for the current task."""
        self.action_node = self.DB.read_one_node(nid)
        return self.action_node

    def set_action_edge(self, src:int, dst:int):
        """Set the action edge for the current task."""
        self.action_edge = self.DB.read_one_edge(src, dst)
        return self.action_edge
    
    def action_stack_push(self, action:Action):
        if action.action_type == 'add_path':
            is_successed = self.add_path(action)
        elif action.action_type == 'delete_nodes':
            # record history for the action
            history = {
                'nodes': {action.action_node['nid']: action.action_node},
                'edges': self.DB.read_edges_by_nids([action.action_node['nid']])
            }
            action.record_history(history)
            is_successed = self.delete_nodes(action, revoke=False)
        elif action.action_type == 'delete_edges':
            # record history for the action
            src, dst = action.action_edge['src'], action.action_edge['dst']
            history = {
                'nodes': self.DB.read_nodes([src, dst]),
                'edges': {(src, dst): action.action_edge}
            }
            action.record_history(history)
            is_successed = self.delete_edges(action, revoke=False)
        elif action.action_type == 'update_node':
            history = {
                'nodes': self.DB.read_nodes([action.task_node['nid']]),
                'edges': {}
            }
            action.record_history(history)
            self.action_stack.append(action)
            is_successed = True
        else:
            raise ValueError(f"Unknown action: {action.action_type}")
        return is_successed
    
    def action_stack_pop(self):
        if len(self.action_stack) == 0:
            return None
        action = self.action_stack.pop()
        if action.action_type == 'add_path':
            self.delete_nodes(action, revoke=True)
            self.delete_edges(action, revoke=True)
        elif action.action_type == 'delete_nodes':
            self._add_nodes(action, revoke=True)
        elif action.action_type == 'delete_edges':
            self._add_edges(action, revoke=True)
        else:
            raise ValueError(f"Unknown action: {action.action_type}")
        return action

    def action_stack_clear(self):
        self.action_stack.clear()
    
    def actions_apply2db(self):
        for action in self.action_stack:
            # update db status
            self.init_db_status()
            if action.action_type == 'add_path':
                added_nodes = {}
                added_edges = {}
                max_nid = self.MAX_NID

                # add new path nodes
                for _nid, node_data in action.path_nodes.items():
                    nid = max_nid + (- _nid)
                    node_data.update({
                        'nid': nid,
                        'creator': action.creator,
                        'type': 0,
                        'checked': 0,
                        'sid': -1,
                    })
                    added_nodes[nid] = node_data
                self.DB.add_nodes(added_nodes)

                # add new path edges
                for (_src, _dst), edge_data in action.path_edges.items():
                    src = max_nid + (- _src)
                    dst = max_nid + (- _dst)
                    edge_data.update({
                        'creator': action.creator,
                    })
                    added_edges[(src, dst)] = edge_data
                self.DB.add_edges(added_edges)

                # check the action node
                self.DB.update_nodes([action.action_node['nid']], creator=action.creator, checked=1)
                self.finish_one_task(action.action_node['nid'])
            
            elif action.action_type == 'delete_nodes':
                # the nodes to be deleted is the action node
                # action node and its edges are also recorded in the action.history
                deleted_nids = [action.action_node['nid']]
                self.DB.delete_nodes(deleted_nids)

                deleted_SrcDst = action.history['edges'].keys()
                self.DB.delete_edges(deleted_SrcDst)
            
            elif action.action_type == 'delete_edges':
                # the edges to be deleted is the action edge
                deleted_SrcDst = [(action.action_edge['src'], action.action_edge['dst'])]
                self.DB.delete_edges(deleted_SrcDst)
            
            elif action.action_type == 'update_node':
                # check the task node
                task_node = action.task_node
                self.DB.update_nodes([task_node['nid']], creator=task_node['creator'], type=task_node['type'], checked=1)
                self.finish_one_task(task_node['nid'])

    def actions_record2db(self):
        self.DB.add_actions(self.action_stack)

    def submit(self):
        self.actions_apply2db()
        self.actions_record2db()
        self.action_stack_clear()
        self.init_db_status()

    def add_path(self, action:Action):
        try:
            self._add_nodes(action, revoke=False)
            self._add_edges(action, revoke=False)
            self.action_stack.append(action)
        except Exception as e:
            print(f"Error adding path: {e}")
            return False
        return True

    def _add_nodes(self, action:Action, *, revoke:bool):
        if revoke:
            self.G.add_nodes(action.history['nodes'])
            self.G.add_edges(action.history['edges'])
        else:
            self.G.add_nodes(action.path_nodes)

    def _add_edges(self, action:Action, *, revoke:bool):
        if revoke:
            self.G.add_edges(action.history['edges'])
        else:
            self.G.add_edges(action.path_edges)

    def delete_nodes(self, action:Action, *, revoke:bool):
        try:
            self.G.delete_nodes(action.path_nodes)
            if not revoke:
                self.action_stack.append(action)
        except Exception as e:
            print(f"Error deleting nodes: {e}")
            return False
        return True
    
    def delete_edges(self, action:Action, *, revoke:bool):
        try:
            self.G.delete_edges(action.path_edges)
            if not revoke:
                self.action_stack.append(action)
        except Exception as e:
            print(f"Error deleting edges: {e}")
            return False
        return True



    