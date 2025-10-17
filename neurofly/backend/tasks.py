class Tasks():
    def __init__(self, tasks:dict=None):
        self.init_global(tasks)
        self.init_dynamic_stack()
        self.reset_idx()

    def init_global(self, tasks:dict=None):
        self.TASKS = {}
        self.UncheckedList = []
        self.CheckedList = []
        if tasks is not None:
            for idx, task in enumerate(tasks):
                self.TASKS[task['nid']] = {
                    'idx': idx,
                    'nid': task['nid'],
                    'checked': task.get('checked', -1),
                    'coord': task['coord'],
                }
                if task.get('checked', -1) == -1:
                    self.UncheckedList.append(task['nid'])
                else:
                    self.CheckedList.append(task['nid'])

    def init_dynamic_stack(self, nid=None):
        if nid is not None:
            self.DynamicStack = [nid]
        else:
            self.DynamicStack = []
    
    def reset_idx(self):
        self.reset_dynamic_idx()
        self.reset_unchecked_idx()
        self.reset_checked_idx()
    
    def reset_dynamic_idx(self):
        self.idx_dynamic = -1

    def reset_unchecked_idx(self):
        self.idx_unchecked = 0
    
    def reset_checked_idx(self):
        self.idx_checked = -1
    
    def add_task_in_dynamicStack(self, nodes:dict):
        for _nid, _node in nodes.items():
            if _nid not in self.DynamicStack:
                self.DynamicStack.append(_nid)
            if _nid not in self.UncheckedList and _nid not in self.TASKS:
                self.UncheckedList.append(_nid)
                self.TASKS[_nid] = {
                    'idx': len(self.UncheckedList) - 1,
                    'nid': _nid, 
                    'checked': -1,
                    'coord': _node['coord'],
                }

    def get_status(self):
        len_unchecked = len(self.UncheckedList)
        len_checked = len(self.CheckedList)
        len_dynamic = len(self.DynamicStack)
        status = {
            'dynamic_length': len_dynamic,
            'unchecked_length': len_unchecked,
            'checked_length': len_checked,
            'idx_dynamic': self.idx_dynamic,
            'idx_unchecked': self.idx_unchecked,
            'idx_checked': self.idx_checked,
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
        if self.UncheckedList:
            idx = self.idx_unchecked % len(self.UncheckedList)
            self.idx_unchecked += 1
            task_nid = self.UncheckedList[idx]
        return task_nid

    def get_task_from_checked(self):
        task_nid = None
        if self.CheckedList:
            idx = self.idx_checked % len(self.CheckedList)
            self.idx_checked -= 1
            task_nid = self.CheckedList[idx]
        return task_nid

    def finish_task(self, nid):
        if nid in self.DynamicStack:
            self.DynamicStack.remove(nid)
        if nid in self.UncheckedList:
            idx = self.UncheckedList.index(nid)
            self.UncheckedList.pop(idx)
            self.TASKS[nid]['checked'] = 1
        self.CheckedList.append(nid)
    
    def remove_task(self, nid):
        if nid in self.DynamicStack:
            self.DynamicStack.remove(nid)
        if nid in self.UncheckedList:
            idx = self.UncheckedList.index(nid)
            self.UncheckedList.pop(idx)
            del self.TASKS[nid]
        if nid in self.CheckedList:
            idx = self.CheckedList.index(nid)
            self.CheckedList.pop(idx)
            del self.TASKS[nid]
