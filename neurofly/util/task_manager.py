from .action import Action
from .neuron_graph import NeuroGraph
from ..neurodb.neurodb_sqlite import NeurodbSQLite

class TaskManager():
    def __init__(self, db_path:str=None):
        self.db_path = db_path
        self.DB = NeurodbSQLite(db_path) if db_path else None
        self.G = NeuroGraph()

        self.action_stack:list[Action] = []

        self.init_task()
    
    def init_task(self):
        self.tasks = self.DB.read_tasks() if self.DB else []
        self.curr_idx = 0
    
    def init_graph(self, roi):
        nodes_coords, edges_coords = self.DB.read_nodes_edges_within_roi(roi)
        self.G.init_graph(nodes_coords, edges_coords)

    def info(self):
        info = {
            'total': len(self.tasks),
            'finished': self.curr_idx,
            'remaining': len(self.tasks) - self.curr_idx
        }
        return info

    def get_one_task(self, idx:int=None):
        if idx is None:
            idx = self.curr_idx
        task = self.tasks[idx]
        nid = task['nid']
        coord = task['coord']
        seg_len = task['seg_len']
        return nid, coord, seg_len

    def get_next_task(self):
        if self.curr_idx >= len(self.tasks):
            return None, None, None
        nid, coord, seg_len = self.get_one_task(self.curr_idx)
        self.curr_idx += 1
        return nid, coord, seg_len
    
    def action_stack_push(self, action:Action):
        if action.action_type == 'add_path':
            is_successed = self.add_path(action)
        elif action.action_type == 'delete_nodes':
            is_successed = self.delete_nodes(action, revoke=False)
        elif action.action_type == 'delete_edges':
            is_successed = self.delete_edges(action, revoke=False)
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
    
    def submit(self):
        self.action_stack_clear()

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
            self.G.add_nodes(action.nodes)

    def _add_edges(self, action:Action, *, revoke:bool):
        if revoke:
            self.G.add_edges(action.history['edges'])
        else:
            self.G.add_edges(action.edges)

    def delete_nodes(self, action:Action, *, revoke:bool):
        try:
            history = self.G.delete_nodes(action.nodes)
            action.record_history(history)
            if not revoke:
                self.action_stack.append(action)
        except Exception as e:
            print(f"Error deleting nodes: {e}")
            return False
        return True
    
    def delete_edges(self, action:Action, *, revoke:bool):
        try:
            history = self.G.delete_edges(action.edges)
            action.record_history(history)
            if not revoke:
                self.action_stack.append(action)
        except Exception as e:
            print(f"Error deleting edges: {e}")
            return False
        return True



    