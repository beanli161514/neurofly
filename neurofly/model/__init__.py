from pathlib import Path
 
# defalut weight paths
package_dir = Path(__file__).resolve().parent

default_seger_weight_path = str(package_dir / 'seg_fiber' / 'universal_tiny.pth')
default_dec_weight_path = str(package_dir / 'resin' / 'mpcn_dumpy.pth')
default_transformer_weight_path = str(package_dir / 'tracer' / 'next_pos.pth')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
if TORCH_AVAILABLE and torch.cuda.is_available():
    from neurofly.model.seg_fiber.segnet_torch import SegNet
    from neurofly.model.resin.mpcn_torch import Deconver
    from neurofly.model.tracer.twoway_transformer import PosPredictor
else:
    from neurofly.model.seg_fiber.segnet_tinygrad import SegNet
    from neurofly.model.resin.mpcn_tinygrad import Deconver

from neurofly.model.seg_fiber.seger import Seger