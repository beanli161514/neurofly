import torch
import os
import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import get_child

class BatchNorm3d:
    def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

        if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
        else: self.weight, self.bias = None, None

        self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
        self.num_batches_tracked = Tensor.zeros(0, requires_grad=False)

    def __call__(self, x:Tensor):
        # for inference only
        batch_invstd = self.running_var.reshape(1, -1, 1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
        bn_init = (x - self.running_mean.reshape(1, -1, 1, 1, 1).expand(x.shape)) * batch_invstd
        return self.weight.reshape(1, -1, 1, 1, 1).expand(x.shape) * bn_init + self.bias.reshape(1, -1, 1, 1, 1).expand(x.shape)


class DownsampleBlock:
    def __init__(self, c0, c1, stride=1):
        self.conv = [nn.Conv2d(c0, c1, kernel_size=(3,3,3), stride=stride, padding=(1,1,1,1,1,1), bias=False), BatchNorm3d(c1), Tensor.relu, nn.Conv2d(c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), BatchNorm3d(c1), Tensor.relu]

    def __call__(self, x):
        return x.sequential(self.conv)

class UpsampleBlock:
    def __init__(self, c0, c1):
        self.conv = [nn.Conv2d(c0, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), BatchNorm3d(c1), Tensor.relu, nn.Conv2d(c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), BatchNorm3d(c1), Tensor.relu]

    def __call__(self, x):
        return x.sequential(self.conv)


class UNet3D:
    def __init__(self, in_channels=1,filters=[32,64,128], n_class=1):
        self.downs = []
        self.ups = []
        for feature in filters:
            self.downs.append(DownsampleBlock(in_channels,feature))
            in_channels = feature
        for feature in reversed(filters[:-1]):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=(2,2,2),stride=2))
            self.ups.append(UpsampleBlock(feature*2,feature))

        self.final_conv = [nn.Conv2d(filters[0], n_class, kernel_size=(1, 1, 1))]


    def __call__(self, x):
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = Tensor.max_pool2d(x,kernel_size=(2,2,2),stride=(2,2,2))

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            x = skip.cat(x, dim=1)
            x = self.ups[i+1](x)
        x = self.final_conv[0](x)
        return x.sigmoid()


    def load_from_pretrained(self,ckpt_path):
        state_dict = torch.load(ckpt_path,map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
            torch.save(state_dict, ckpt_path)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        for k, v in state_dict.items():
            obj = get_child(self, k)
            if obj.shape == v.shape:
                obj.assign(v.numpy())
            else:
                pass


class SegNet():
    def __init__(self,ckpt_path,bg_thres=150):
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
        model = UNet3D(1, model_dims, 1)
        model.load_from_pretrained(ckpt_path)

        self.model = model
        self.bg_thres = bg_thres
    
    def preprocess(self,img):
        # input img nparray [0,65535]
        # output img tensor [0,1]
        d,w,h = img.shape
        max = img.max()
        min = img.min()
        img = (img-min)/(max-min)
        img = img.astype(np.float32)
        img = Tensor(img,requires_grad=False)
        img = img.reshape(1,1,d,w,h)
        return img


    def get_mask(self,img,thres=0.5):
        d,w,h = img.shape
        img_in = self.preprocess(img)
        if img_in is not None:
            tensor_out = self.model(img_in)
            prob = tensor_out.reshape(d,w,h)
            if thres==None:
                return prob.numpy()
            else:
                voxel_mask = img > self.bg_thres
                prob = prob.numpy()
                prob[prob>=thres]=1
                prob[prob<thres]=0
                return prob*voxel_mask
        else:
            return np.zeros_like(img)

