class Action():
    def __init__(self, action_type:str, nodes:dict=None, edges:dict=None):
        self.ACTION_TYPE_LIST = ['add_path', 'delete_nodes', 'delete_edges']
        if action_type in self.ACTION_TYPE_LIST:
            self.action_type = action_type
        else:
            raise Exception("Invalid action type!")
        self.task_nid = None
        self.task_coord = None
        self.nodes = nodes
        self.edges = edges
        self.history = None

    def record_history(self, history):
        """Record the history of the action."""
        self.history = history