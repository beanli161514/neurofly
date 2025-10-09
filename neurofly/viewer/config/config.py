import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_default.yaml')

class NeuronViewerConfig:
    ROISelector: dict = {
        'max_size': 512,
        'default_center': (64, 64, 64),
        'default_size': (64, 64, 64),
        'power_step': 32
    }
    auto_contrast: bool = True
    def __init__(self, config:dict):
        self.cfg = config if config else {}
        self.init_common_cfg()
        self.init_roi_cfg()
    
    def init_common_cfg(self):
        self.auto_contrast = self.cfg.get('auto_contrast', True)
    
    def init_roi_cfg(self):
        roi_cfg = self.cfg.get('ROISelector', {})
        for key, value in roi_cfg.items():
            if key in self.ROISelector:
                self.ROISelector[key] = value
        
class NeuronSegmenterConfig:
    model_ckpt_path: str
    def __init__(self, config:dict):
        self.cfg = config if config else {}
        self.model_ckpt_path = config.get('model_ckpt_path', '')

class NeuronReconstructorConfig:
    global_task_query: str
    auto_contrast: bool
    DFS: bool
    def __init__(self, config:dict):
        self.cfg = config if config else {}
        self.global_task_query = config.get('global_task_query', '')
        self.auto_contrast = config.get('auto_contrast', True)
        self.DFS = config.get('DFS', True)

class Config:
    def __init__(self, config_path:str=None):
        if config_path is None:
            user_home = os.path.expanduser('~')
            config_path = os.path.join(user_home, '.config/' 'neurofly_config.yaml')
        if not os.path.exists(config_path):
            config_path = DEFAULT_CONFIG_PATH
        print(f'Using config file: {config_path}')
        self.read_cfg(config_path)

        self.init_viewer_cfg()
        self.init_segmenter_cfg()
        self.init_reconstructor_cfg()

    def read_cfg(self, config_path:str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        return self.cfg
    
    def init_viewer_cfg(self):
        self.viewer_cfg = NeuronViewerConfig(self.cfg.get('NeuronViewer', {}))
    
    def init_segmenter_cfg(self):
        self.segmenter_cfg = NeuronSegmenterConfig(self.cfg.get('NeuronSegmenter', {}))
    
    def init_reconstructor_cfg(self):
        self.reconstructor_cfg = NeuronReconstructorConfig(self.cfg.get('NeuronReconstructor', {}))