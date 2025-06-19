import torch
import numpy as np
import torch.nn as nn
import functools


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
    def __init__(self, in_channels, out_channels, *, norm_type=None, dim=3):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        if norm_type is not None:
            norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        conv_layers = []
        conv_layers.append(Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        if norm_type is not None:
            conv_layers.append(norm_layer(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        if norm_type is not None:
            conv_layers.append(norm_layer(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*conv_layers)


    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_3d_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], *, norm_type='batch', dim=3):
        super(UNet_3d_Generator, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
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
                nn.Sequential(
                    upsample,
                    Conv(feature*2, feature, kernel_size=1, stride=1)
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = Conv(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        input = x
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
        return x + input


class Deconver():
    def __init__(self,weight_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)
        weight_dict = torch.load(weight_path, weights_only=False)
        if 'params' in weight_dict.keys():
            weight_dict = weight_dict['params']
            torch.save(weight_dict, weight_path)
        model.load_state_dict(weight_dict, strict=True)
        model.eval()
        model.to(self.device)
        self.model = model


    def preprocess(self,img,percentiles=[0,0.9999]):
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr) 
        img = (clipped_arr-min_value)/(max_value-min_value)
        return img, min_value, max_value


    def postprocess(self,img,min_value,max_value):
        img = img.detach().cpu().numpy()
        img = img * (max_value - min_value) + min_value
        img = np.clip(img, 0, 65535)
        return img.squeeze()
    

    def process_one(self,img):
        img, min_value, max_value = self.preprocess(img, percentiles=[0, 0.9999])
        img = img.astype(np.float32)[None, None,]
        img = torch.from_numpy(img).to(self.device)
        out = self.model(img)
        sr_img = self.postprocess(out,min_value,max_value).astype(np.uint16)
        return sr_img
