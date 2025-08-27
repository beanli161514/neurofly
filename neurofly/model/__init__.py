import torch
from pathlib import Path
 
# defalut weight paths
package_dir = Path(__file__).resolve().parent

default_seger_weight_path = str(package_dir / 'seg_fiber' / 'universal_tiny.pth')
default_dec_weight_path = str(package_dir / 'resin' / 'mpcn_dumpy.pth')
default_transformer_weight_path = str(package_dir / 'tracer' / 'next_pos.pth')

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
if torch.cuda.is_available():
    from .seg_fiber.segnet_torch import SegNet
    from .resin.mpcn_torch import Deconver
else:
    from .seg_fiber.segnet_tinygrad import SegNet
    from .resin.mpcn_tinygrad import Deconver

from .tracer.twoway_transformer import PosPredictor
from .seg_fiber.seger import Seger