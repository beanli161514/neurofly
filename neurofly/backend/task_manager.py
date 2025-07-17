from .action import Action
from .neuron_graph import NeuroGraph
from ..neurodb.neurodb_sqlite import NeurodbSQLite

class TaskManager():
    def __init__(self, db_path:str=None):
        self.db_path = db_path
        self.DB = NeurodbSQLite(db_path)
        self.G = NeuroGraph()

        self.action_stack:list[Action] = []

        self.init_task()
        self.reset_task_status()
    
    def init_graph(self, roi):
        nodes_coords, edges_coords = self.DB.read_nodes_edges_within_roi(roi)
        self.G.init_graph(nodes_coords, edges_coords)

    def init_db_status(self):
        self.MAX_NID = self.DB.get_max_nid()
        self.MAX_SID, _ = self.DB.get_max_sid_version()
    
    def init_task(self):
        tasks = self.DB.read_tasks() if self.DB else []
        self.TASKS = {
            'tasks': {},
            'checked_list': [],
            'unchecked_list': [],
            'last_idx': -1,
            'next_idx': 0,
            'last_submit': False,
        }
        for idx, task in enumerate(tasks):
            self.TASKS['unchecked_list'].append(task['nid'])
            self.TASKS['tasks'][task['nid']] = {
                'idx': idx,
                'nid': task['nid'],
                'checked': -1,
                'coord': task['coord'],
                'cnnt_len': task['cnnt_len']
            }
    
    def reset_task_status(self):
        self.TASKS.update({
            'last_submit': True,
            'next_idx': 0,
            'last_idx': -1,
        })
        self.task_node = None
        self.action_node = None
        self.action_edge = None

    def get_next_task(self):
        if len(self.TASKS['unchecked_list']) == 0:
            return None
        else:
            next_idx = self.TASKS['next_idx'] % len(self.TASKS['unchecked_list'])
            task_nid = self.TASKS['unchecked_list'][next_idx]
            task = self.TASKS['tasks'][task_nid]
            self.set_task_node(task_nid)
            self.TASKS['next_idx'] = next_idx + 1
            return task
    
    def get_last_task(self):
        if len(self.TASKS['checked_list']) == 0:
            return None
        else:
            last_idx = self.TASKS['last_idx'] % len(self.TASKS['checked_list'])
            task_nid = self.TASKS['checked_list'][last_idx]
            task = self.TASKS['tasks'][task_nid]
            self.set_task_node(task_nid)
            self.TASKS['last_idx'] = last_idx-1
            return task
            
    def finish_one_task(self, nid:int):
        if nid in self.TASKS['unchecked_list']:
            idx = self.TASKS['unchecked_list'].index(nid)
            self.TASKS['unchecked_list'].pop(idx)
            self.TASKS['checked_list'].append(nid)
            self.TASKS['tasks'][nid]['checked'] = 1
        
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
        try:
            if action.action_type == 'add_path':
                self.G.add_nodes(action.path_nodes)
                self.G.add_edges(action.path_edges)
            elif action.action_type == 'delete_node':
                history = {
                    'nodes': {action.action_node['nid']: action.action_node},
                    'edges': self.DB.read_edges_by_nids([action.action_node['nid']])
                }
                # record history for the action
                action.record_history(history)
                self.G.delete_nodes({action.action_node['nid']: action.action_node})
            elif action.action_type == 'delete_edge':
                src, dst = action.action_edge['src'], action.action_edge['dst']
                history = {
                    'nodes': self.DB.read_nodes([src, dst]),
                    'edges': {(src, dst): action.action_edge}
                }
                # record history for the action
                action.record_history(history)
                src_dst = (action.action_edge['src'], action.action_edge['dst'])
                self.G.delete_edges({src_dst: action.action_edge})
            elif action.action_type == 'update_node':
                history = {
                    'nodes': self.DB.read_nodes([action.task_node['nid']]),
                    'edges': {}
                }
                action.record_history(history)
            else:
                raise ValueError(f"Unknown action: {action.action_type}")
            is_successed = True
            self.action_stack.append(action)
        except Exception as E:
            print(f'Error in action_stack_push: {E}')
            is_successed = False
        finally:
            return is_successed
    
    def action_stack_pop(self):
        if len(self.action_stack) == 0:
            return None
        try:
            action = self.action_stack.pop()
            if action.action_type == 'add_path':
                self.G.delete_nodes(action.path_nodes)
                self.G.delete_edges(action.path_edges)
            elif action.action_type == 'delete_node':
                self.G.add_nodes(action.history['nodes'])
                self.G.add_edges(action.history['edges'])
            elif action.action_type == 'delete_edge':
                self.G.add_edges(action.history['edges'])
            else:
                raise ValueError(f"Unknown action: {action.action_type}")
        except Exception as E:
            print(f'Error in action_stack_pop: {E}')
            self.action_stack.append(action)
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
                _neg_nid_offset = min(action.path_nodes.keys())
                for _neg_temp_nid, node_data in action.path_nodes.items():
                    nid = max_nid + (- (_neg_temp_nid - _neg_nid_offset) + 1)
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
                for (_neg_temp_src_nid, _neg_temp_dst_nid), edge_data in action.path_edges.items():
                    src = max_nid + (- (_neg_temp_src_nid - _neg_nid_offset) + 1)
                    dst = max_nid + (- (_neg_temp_dst_nid - _neg_nid_offset) + 1)
                    edge_data.update({
                        'creator': action.creator,
                    })
                    added_edges[(src, dst)] = edge_data
                self.DB.add_edges(added_edges)

                # check the action node
                if action.action_node['nid'] >0 :
                    self.DB.update_nodes([action.action_node['nid']], creator=action.creator, checked=1)
                    self.finish_one_task(action.action_node['nid'])
            
            elif action.action_type == 'delete_node':
                # the nodes to be deleted is the action node
                # action node and its edges are also recorded in the action.history
                deleted_nids = [action.action_node['nid']]
                self.DB.delete_nodes(deleted_nids)

                deleted_SrcDst = action.history['edges'].keys()
                self.DB.delete_edges(deleted_SrcDst)
            
            elif action.action_type == 'delete_edge':
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
        self.reset_task_status()
        
        