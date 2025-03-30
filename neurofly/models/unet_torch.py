import torch.nn as nn
import functools
import torch
import numpy as np


def get_norm_layer(norm_type='instance', dim=2):
    if dim == 2:
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    elif dim == 3:
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    else:
        raise Exception('Invalid dim.')
    
    if norm_type == 'batch':
        norm_layer = functools.partial(BatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm, affine=False, track_running_stats=False)
    elif norm_type == 'identity':
        def norm_layer(x):
            return lambda t:t
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type='batch', dim=2):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], *, norm_type='batch', dim=2):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            self.MaxPool = nn.MaxPool3d
        else:
            raise Exception('Invalid dim.')

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, norm_type=norm_type, dim=dim))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                ConvTranspose(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = nn.Sequential(
            Conv(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = self.MaxPool(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        return x


class SegNet():
    def __init__(self,ckpt_path,bg_thres=150):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'tiny' in ckpt_path:
            model_dims = [32,64,128]
        elif 'medium' in ckpt_path:
            model_dims = [32,64,128,256]
        elif 'dumpy' in ckpt_path:
            model_dims = [64,128,256]
        model = UNet(1, 1, model_dims, norm_type='batch', dim=3)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt.keys():
            model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
            torch.save(ckpt['model'], ckpt_path)
        else:
            model.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()})
        model.to(self.device)
        model.eval()
        self.model = model
        self.bg_thres = bg_thres

    
    def preprocess(self,img):
        # input img: ndarray [0,65535]
        # output img: tensor [0,1]
        min_value = img.min()
        max_value = img.max()
        img = (img-min_value)/(max_value-min_value)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        return img


    def get_mask(self,img,thres=0.5):
        img_in = self.preprocess(img)
        if img_in != None:
            with torch.no_grad():
                tensor_out = self.model(img_in.to(self.device)).cpu()
            prob = tensor_out.squeeze(0).squeeze(0)
            if thres==None:
                return prob.detach().numpy()
            else:
                voxel_mask = img > self.bg_thres
                prob[prob>=thres]=1
                prob[prob<thres]=0
                return prob.detach().numpy()*voxel_mask
        else:
            return np.zeros_like(img)
