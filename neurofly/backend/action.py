class Action():
    def __init__(self, creator:str, task_node:dict, action_type:str, *, 
                 action_node:dict=None, action_edge:dict=None,
                 path_nodes:dict={}, path_edges:dict={}):
        self.creator = creator
        self.task_node = task_node
        self.action_node = action_node
        self.action_edge = action_edge

        self.ACTION_TYPE_LIST = ['add_path', 'delete_node', 'delete_edge', 'update_node', 'deconv']
        if action_type in self.ACTION_TYPE_LIST:
            self.action_type = action_type
        else:
            raise Exception("Invalid action type!")
        
        self.record_path_nodes(path_nodes)
        self.record_path_edges(path_edges)
        self.history = {
            'nodes': {},
            'edges': {}
        }

    def record_path_nodes(self, path_nodes:dict):
        for nid in path_nodes.keys():
            path_nodes[nid]['creator'] = self.creator
        self.path_nodes = path_nodes
    
    def record_path_edges(self, path_edges:dict):
        for (src, dst) in path_edges.keys():
            path_edges[(src, dst)]['creator'] = self.creator
        self.path_edges = path_edges
        
    def record_history(self, history:dict):
        """Record the history of the action."""
        self.history = history
            