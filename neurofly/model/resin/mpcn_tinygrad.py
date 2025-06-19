import torch
import os
import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import get_child


class DownsampleBlock:
    def __init__(self, c0, c1, stride=1):
        self.conv = [
            nn.Conv2d(c0, c1, kernel_size=(3,3,3), stride=stride, padding=(1,1,1,1,1,1), bias=False),
            Tensor.relu,
            nn.Conv2d(c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False),
            Tensor.relu
            ]

    def __call__(self, x):
        return x.sequential(self.conv)


class MPCN:
    def __init__(self,in_channels, filters, n_class):

        self.downs = []
        self.ups = []

        for feature in filters:
            self.downs.append(DownsampleBlock(in_channels,feature))
            in_channels = feature
        for feature in reversed(filters[:-1]):
            self.ups.append(nn.Conv2d(feature*2, feature, kernel_size=(1,1,1),stride=1))
            self.ups.append(DownsampleBlock(feature*2,feature))

        self.final_conv = nn.Conv2d(filters[0], n_class, kernel_size=(1, 1, 1), bias=True)


    def __call__(self, x):
        input = x
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = Tensor.max_pool2d(x,kernel_size=(2,2,2),stride=(2,2,2))

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            _, _, D, H, W = x.shape
            x = x.interpolate((D*2,H*2,W*2), mode='linear')
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            x = skip.cat(x, dim=1)
            x = self.ups[i+1](x)
        x = self.final_conv(x)
        return x + input


    def load_from_pretrained(self,ckpt_path):
        state_dict = torch.load(ckpt_path,map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
            torch.save(state_dict, ckpt_path)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}

        for k, v in state_dict.items():
            if 'ups' in k:
                k = k.split('.')
                if len(k) == 4:
                    k.pop(2)
                k = '.'.join(k)
            obj = get_child(self, k)
            if obj.shape == v.shape:
                obj.assign(v.numpy())
            else:
                pass


class Deconver():
    def __init__(self,ckpt_path):
        # TODO: remove this after conflicts with conda were solved
        os.environ['METAL_XCODE'] = '1'
        os.environ['DISABLE_COMPILER_CACHE'] = '1'
        os.environ['GPU'] = '1'
        if 'tiny' in ckpt_path:
            model_dims = [32,64,128]
        elif 'medium' in ckpt_path:
            model_dims = [32,64,128,256]
        elif 'dumpy' in ckpt_path:
            model_dims = [64,128,256]
        model = MPCN(1, model_dims, 1)
        model.load_from_pretrained(ckpt_path)
        self.model = model


    def preprocess(self, img, percentiles=[0,0.9999]):
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

        min_value = clipped_arr.min()
        max_value = clipped_arr.max()
        img = (img-min_value)/(max_value-min_value)
        img = img[None,None,]
        return img, min_value, max_value


    def postprocess(self,img,min_value,max_value):
        img = img
        img = img * (max_value - min_value) + min_value
        img = np.clip(img, 0, 65535)
        return img.squeeze()
    

    def process_one(self,img):
        img = img.astype(np.float32)
        img, min_value, max_value = self.preprocess(img)
        img = Tensor(img,requires_grad=False)
        out = self.model(img).numpy()
        sr_img = self.postprocess(out,min_value,max_value).astype(np.uint16)
        return sr_img

