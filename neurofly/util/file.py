import sys


class Tee(object):
    def __init__(self, file_path:str, mode='w', keep_stdout:bool=True, keep_stderr:bool=False):
        file = open(file_path, mode)
        self.files = [file]
        if keep_stdout:
            self.files.append(sys.__stdout__)
        if keep_stderr:
            self.files.append(sys.__stderr__)
    
    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()