import os
import json
import pprint

from neurofly.backend.action import Action
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.backend.tasks import Tasks
from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.viewer.config.config import Config

class TaskManager():
    def __init__(self, db_path:str, config:Config=None):
        self.db_path = db_path
        self.config = config
        self.DB = NeurodbSQLite(db_path)
        self.G = NeuroGraph()
        self.G_prof = NeuroGraph()

        self.action_stack:list[Action] = []

        self.init_task()
        self.reset_task_status()
    
    def init_graph(self, roi):
        nodes, edges = self.DB.read_nodes_edges_within_roi(roi)
        self.G.init_graph(nodes, edges)
    
    def init_graph_prof(self):
        if self.task_node is not None:
            nodes, edges = self.DB.read_connected_components(self.task_node['nid'], with_edges=True)
            self.G_prof.init_graph(nodes, edges)

    def init_db_status(self):
        self.MAX_NID = self.DB.get_max_nid()
        self.MAX_SID, _ = self.DB.get_max_sid_version()
    
    def init_task(self):
        if self.DB:
            global_task_query = ''
            # use the global_task_query in config if available
            if self.config:
                global_task_query = self.config.reconstructor_cfg.global_task_query
            if global_task_query.endswith('.json') and os.path.exists(global_task_query):
                tasks = self.read_task_from_json(global_task_query)
                pprint.pprint(f'Loaded {len(tasks)} tasks from json file: {global_task_query}')
            else:
                tasks = self.DB.read_tasks(global_task_query)
                pprint.pprint(f'Loaded {len(tasks)} tasks from db with query: {global_task_query}')
        else:
            tasks = []
        self.TASKS = Tasks(tasks)

    def read_task_from_json(self, json_path:str):
        with open(json_path, 'r') as f:
            data = json.load(f)
            candidate_tasks = data['tasks']
        candidate_task_nids = [task['nid'] for task in candidate_tasks]
        valid_nodes = self.DB.read_nodes(candidate_task_nids)
        tasks = []
        for nid, node in valid_nodes.items():
            tasks.append({
                'nid': nid,
                'coord': node['coord'],
                'checked': node['checked'],
            })
        return tasks
    
    def reset_task_status(self):
        self.TASKS.reset_idx()
        self.task_node = None
        self.action_node = None
        self.action_edge = None

    def get_task_status(self):
        return self.TASKS.get_status()

    def get_next_task(self):
        # use DFS to get the next task from dynamic stack if possible
        if self.config.reconstructor_cfg.DFS:
            task_nid_dynamic = self.TASKS.get_task_from_dynamicStack()
            if task_nid_dynamic is not None:
                task_node = self.set_task_node(task_nid_dynamic)
            else:
                # if dynamic stack is empty, get a task from unchecked list and add it to dynamic stack
                # but not set it as task node in this round
                task_node = None
                task_nid_unchecked = self.TASKS.get_task_from_unchecked()
                if task_nid_unchecked is not None:
                    self.TASKS.add_task_in_dynamicStack({task_nid_unchecked: self.TASKS.TASKS[task_nid_unchecked]})
        # just get the next task from unchecked list
        else:
            task_nid_unchecked = self.TASKS.get_task_from_unchecked()
            task_node = self.set_task_node(task_nid_unchecked)
        self.TASKS.reset_checked_idx()
        return task_node
    
    def get_last_task(self):
        task_nid = self.TASKS.get_task_from_checked()
        if task_nid is not None:
            task_node = self.set_task_node(task_nid, init_dynamic_stack=False)
            self.TASKS.reset_unchecked_idx()
        else:
            task_node = None
        return task_node
            
    def finish_task(self, nid:int):
        self.TASKS.finish_task(nid)
        
    def set_task_node(self, nid:int, *, init_dynamic_stack:bool=True):
        """Set the task node for the current task."""
        if nid is None:
            self.task_node = None
        else:
            self.task_node = self.DB.read_one_node(nid)
            if self.config.reconstructor_cfg.DFS and init_dynamic_stack and nid not in self.TASKS.DynamicStack:
                self.TASKS.init_dynamic_stack(nid)
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
                    'nodes': {action.task_node['nid']: action.task_node},
                    'edges': {}
                }
                action.record_history(history)
            elif action.action_type == 'deconv':
                pass
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
                neg_temp_max_nid = self.G.get_neg_temp_max_nid()
                neg_temp_max_nid -= len(action.path_nodes)
                self.G.set_neg_temp_max_nid(neg_temp_max_nid)
            elif action.action_type == 'delete_node':
                self.G.add_nodes(action.history['nodes'])
                self.G.add_edges(action.history['edges'])
            elif action.action_type == 'delete_edge':
                self.G.add_edges(action.history['edges'])
            elif action.action_type == 'update_node':
                pass
            elif action.action_type == 'deconv':
                pass
            else:
                raise ValueError(f"Unknown action: {action.action_type}")
        except Exception as E:
            print(f'Error in action_stack_pop: {E}')
            self.action_stack.append(action)
        return action

    def action_stack_clear(self):
        self.action_stack.clear()

    def update_taskTree_from_action(self, action:Action=None):
        action_nid = action.action_node['nid']
        if action.action_type == 'add_path' and action_nid > 0:
            cc_unchecked_nodes = self.DB.read_unchecked_nodes_in_cc(action_nid)
            if action_nid in cc_unchecked_nodes:
                # let action_nid be the first node in the unchecked nodes
                action_unchecked_node = cc_unchecked_nodes.pop(action_nid)
                cc_unchecked_nodes[action_nid] = action_unchecked_node
            self.TASKS.add_task_in_dynamicStack(cc_unchecked_nodes)
    
    def actions_apply2db(self):
        for a_idx, action in enumerate(self.action_stack):
            # update db status
            self.init_db_status()
            if action.action_type == 'add_path':
                added_nodes = {}
                added_edges = {}
                max_nid = self.MAX_NID

                # add new path nodes
                _neg_nid_offset = max(action.path_nodes.keys()) if len(action.path_nodes) > 0 else 0
                for _neg_temp_nid, node_data in action.path_nodes.items():
                    nid = max_nid + (- (_neg_temp_nid - _neg_nid_offset) + 1)
                    node_data.update({
                        'nid': nid,
                        'creator': action.creator,
                        'type': 0,
                        'checked': node_data.get('checked', 0),
                        'sid': -1,
                    })
                    if node_data['checked'] == -1:
                        self.action_stack[a_idx].action_node['nid'] = nid
                    added_nodes[nid] = node_data
                self.DB.add_nodes(added_nodes)
                if self.config.reconstructor_cfg.DFS:
                    self.update_taskTree_from_action(action)

                # add new path edges
                for (src, dst), edge_data in action.path_edges.items():
                    if src < 0:
                        src = max_nid + (- (src - _neg_nid_offset) + 1)
                    if dst < 0:
                        dst = max_nid + (- (dst - _neg_nid_offset) + 1)
                    edge_data.update({
                        'creator': action.creator,
                    })
                    added_edges[(src, dst)] = edge_data
                self.DB.add_edges(added_edges)
            
            elif action.action_type == 'delete_node':
                # the nodes to be deleted is the action node
                # action node and its edges are also recorded in the action.history
                deleted_nids = [action.action_node['nid']]
                self.DB.delete_nodes(deleted_nids)
                self.TASKS.remove_task(deleted_nids[0])

                deleted_SrcDst = action.history['edges'].keys()
                self.DB.delete_edges(deleted_SrcDst)
            
            elif action.action_type == 'delete_edge':
                # the edges to be deleted is the action edge
                deleted_SrcDst = [(action.action_edge['src'], action.action_edge['dst'])]
                self.DB.delete_edges(deleted_SrcDst)
            
            elif action.action_type == 'update_node':
                # check the task node (== action_node)
                action.action_node.update({'checked': 1})
                action_node = action.action_node
                self.DB.update_nodes([action_node['nid']], creator=action_node['creator'], type=action_node['type'], checked=1)
                self.finish_task(action_node['nid'])
            
            elif action.action_type == 'deconv':
                pass

            else:
                raise ValueError(f"Unknown action: {action.action_type}")

    def actions_record2db(self):
        self.DB.add_actions(self.action_stack)

    def submit(self):
        self.actions_apply2db()
        self.actions_record2db()
        self.action_stack_clear()
        self.init_db_status()
        self.reset_task_status()
        
        