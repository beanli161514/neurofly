class Tasks():
    def __init__(self, tasks:dict=None):
        self.init_global(tasks)
        self.init_dynamic_stack()
        self.reset_idx()

    def init_global(self, tasks:dict=None):
        self.TASKS = {}
        self.Unchecked = []
        self.Checked = []
        if tasks is not None:
            for idx, task in enumerate(tasks):
                self.Unchecked.append(task['nid'])
                self.TASKS[task['nid']] = {
                    'idx': idx,
                    'nid': task['nid'],
                    'checked': -1,
                    'coord': task['coord'],
                    'cnnt_len': task['cnnt_len']
                }

    def init_dynamic_stack(self, nid=None):
        if nid is not None:
            self.DynamicStack = [nid]
        else:
            self.DynamicStack = []
    
    def reset_idx(self):
        self.idx_unchecked = 0
        self.idx_checked = -1
        self.idx_dynamic = -1
    
    def add_task_in_dynamicStack(self, nodes:dict):
        for _nid, _node in nodes.items():
            if _nid not in self.DynamicStack:
                self.DynamicStack.append(_nid)
            if _nid not in self.Unchecked and _nid not in self.TASKS:
                self.Unchecked.append(_nid)
                self.TASKS[_nid] = {
                    'idx': len(self.Unchecked) - 1,
                    'nid': _nid, 
                    'checked': -1,
                    'coord': _node['coord'],
                    'cnnt_len': _node.get('cnnt_len', -1)
                }

    def get_status(self):
        len_unchecked = len(self.Unchecked)
        len_checked = len(self.Checked)
        len_dynamic = len(self.DynamicStack)
        status = {
            'unchecked_length': len_unchecked,
            'checked_length': len_checked,
            'dynamic_length': len_dynamic
        }
        return status

    def get_task_from_dynamicStack(self):
        task_nid = None
        if self.DynamicStack:
            # depth-first search to get the last task
            idx = max(len(self.DynamicStack) + self.idx_dynamic, 0) % len(self.DynamicStack)
            self.idx_dynamic -= 1
            task_nid = self.DynamicStack[idx]
        return task_nid

    def get_task_from_unchecked(self):
        task_nid = None
        if self.Unchecked:
            idx = self.idx_unchecked % len(self.Unchecked)
            self.idx_unchecked += 1
            task_nid = self.Unchecked[idx]
        return task_nid

    def get_task_from_checked(self):
        task_nid = None
        if self.Checked:
            idx = len(self.Checked) + self.idx_checked % len(self.Checked)
            self.idx_checked -= 1
            task_nid = self.Checked[idx]
        return task_nid

    def finish_task(self, nid):
        if nid in self.DynamicStack:
            self.DynamicStack.remove(nid)
        if nid in self.Unchecked:
            idx = self.Unchecked.index(nid)
            self.Unchecked.pop(idx)
            self.Checked.append(nid)
            self.TASKS[nid]['checked'] = 1
