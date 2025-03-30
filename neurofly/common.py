import os
import torch

# defalut weight paths
package_dir = os.path.dirname(os.path.abspath(__file__))
default_seger_weight_path = os.path.join(package_dir,'models/universal_tiny.pth')
default_dec_weight_path = os.path.join(package_dir,'models/mpcn_dumpy.pth')
default_transformer_weight_path = os.path.join(package_dir,'models/next_pos.pth')

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
if torch.cuda.is_available():
    from neurofly.models.unet_torch import SegNet
    from neurofly.models.mpcn_torch import Deconver
else:
    from neurofly.models.unet_tinygrad import SegNet
    from neurofly.models.mpcn_tinygrad import Deconver

from neurofly.models.twoway_transformer import PosPredictor