import os
import torch

# defalut weight paths
package_dir = os.path.dirname(os.path.abspath(__file__))
default_seger_weight_path = os.path.join(package_dir, 'seg_fiber/universal_tiny.pth')
default_dec_weight_path = os.path.join(package_dir, 'resin/mpcn_dumpy.pth')
default_transformer_weight_path = os.path.join(package_dir, 'tracer/next_pos.pth')

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
if torch.cuda.is_available():
    from .seg_fiber.unet_torch import SegNet
    from .resin.mpcn_torch import Deconver
else:
    from .seg_fiber.unet_tinygrad import SegNet
    from .resin.mpcn_tinygrad import Deconver

from .tracer.twoway_transformer import PosPredictor