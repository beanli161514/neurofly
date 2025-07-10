import numpy as np
import h5py
import zarr
import re
from tifffile import imread


class Ims():
    '''
    ims image: [z,y,x]
    input roi and returned image: [x,y,z]
    '''
    def __init__(self,ims_path):
        self.hdf = h5py.File(ims_path,'r')
        self.rois = []
        self.info = self.get_info()
        for i in self.info:
            self.rois.append([int(j/k) for j,k in zip(i['origin'],i['spacing'])] + i['image_size'])
        self.extension = self.info[0]['origin'] + self.info[0]['dims_physical']
        self.dataset = self.hdf.get('DataSet')
        self.time_point_key = 'TimePoint 0'
        self.resolution_levels = list(self.dataset.keys())
        self.channels = self.list_channels(self.resolution_levels[0])


    def list_channels(self, resolution_level):
        res_level = self.dataset.get(resolution_level)
        if res_level is None:
            raise KeyError(f"Resolution level '{resolution_level}' not found.")
        
        time_point_group = res_level.get(self.time_point_key)
        if time_point_group is None:
            raise KeyError(f"Time point '{self.time_point_key}' not found in resolution level '{resolution_level}'.")
        
        return list(time_point_group.keys())


    def __getitem__(self, indices, level=0, channel=0):
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.rois[level][0],x_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        z_slice = slice(z_min-self.rois[level][2],z_max-self.rois[level][2])
        image = self.dataset[self.resolution_levels[level]][self.time_point_key][self.channels[channel]]['Data']
        return np.transpose(image[z_slice,y_slice,x_slice],(2,1,0))


    def from_roi(self, coords, level=0, channel=0, padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        if isinstance(level,str):
            level = self.resolution_levels.index(level)
        if isinstance(channel,str):
            channel = self.channels.index(channel)

        coords = [int(coord) for coord in coords]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.rois[level][0:3]
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.rois[level][0]+xlp,x_max-self.rois[level][0]-xhp)
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        z_slice = slice(z_min-self.rois[level][2]+zlp,z_max-self.rois[level][2]-zhp)
        if isinstance(level, int) and isinstance(channel, int):
            image = self.dataset[self.resolution_levels[level]][self.time_point_key][self.channels[channel]]['Data']
        else:
            return
        img = np.transpose(image[z_slice,y_slice,x_slice],(2,1,0))
        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), padding)
        return padded
    

    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[2],data.shape[1],data.shape[0]],
                    'data_chunks':[data.chunks[2],data.chunks[1],data.chunks[0]],
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info



class ZipZarr():
    '''
    Load hierachical image data of several resolution levels like:
        ├── 1um uint16
        ├── 2um uint16
        ├── 4um uint16
        ├── 8um uint16
        └── 16um uint16
    '''
    def __init__(self,image_path):
        self.store = zarr.open(image_path,mode='r')
        if 'nm' in list(self.store.keys())[0]:
            self.store = self.store['488nm_10X']
        resolution_dict = {
            '1um': [],
            '2um': [],
            '4um': [],
            '8um': [],
            '16um': []
        }
        resolutions = [int(re.findall(r'\d+', dataset_name)[0]) for dataset_name in resolution_dict.keys()]
        for dataset in self.store:
            if '1um' in dataset:
                resolution_dict['1um'] = dataset
            elif '2um' in dataset:
                resolution_dict['2um'] = dataset
            elif '4um' in dataset:
                resolution_dict['4um'] = dataset
            elif '8um' in dataset:
                resolution_dict['8um'] = dataset
            elif '16um' in dataset:
                resolution_dict['16um'] = dataset
        self.images = [self.store[dataset] for key, dataset in resolution_dict.items() if dataset != []]
        self.roi = [0,0,0] + list(self.images[0].shape)
        self.info = []
        self.rois = []
        for i,image in enumerate(self.images):
            self.info.append(
                {
                    'level': i,
                    'spacing': [resolutions[i],resolutions[i],resolutions[i]],
                    'image_size': list(image.shape)
                }
            )
            self.rois.append([0,0,0]+list(image.shape))


    def __getitem__(self, indices, level=0):
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.rois[level][0],x_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        z_slice = slice(z_min-self.rois[level][2],z_max-self.rois[level][2])
        return np.transpose(self.images[level][z_slice,y_slice,x_slice],(2,1,0))


    def from_roi(self, coords, level=0, channel=0,padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        coords = [int(coord) for coord in coords]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.rois[level][0:3] 
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.rois[level][0]+xlp,x_max-self.rois[level][0]-xhp)
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        z_slice = slice(z_min-self.rois[level][2]+zlp,z_max-self.rois[level][2]-zhp) 
        img = np.transpose(self.images[level][z_slice,y_slice,x_slice],(2,1,0))

        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), padding)

        return padded


    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[2],data.shape[1],data.shape[0]],
                    'data_chunks':[data.chunks[2],data.chunks[1],data.chunks[0]],
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info



class Tiff():
    '''
    zarr.attrs['roi'] = [x_offset,y_offset,z_offset,x_size,y_size,z_size]
    To load image directly from global coordinates, wrap .zarr object in this class.
    '''
    def __init__(self,tiff_path):
        self.image = np.squeeze(imread(tiff_path))
        self.roi = [0,0,0] + list(self.image.shape)
        self.rois = [self.roi]
        self.shape = self.roi[3:6]
    
    def __getitem__(self, indices):
        coords = [int(coord) for coord in coords]
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.roi[0],x_max-self.roi[0])
        y_slice = slice(y_min-self.roi[1],y_max-self.roi[1])
        z_slice = slice(z_min-self.roi[2],z_max-self.roi[2])
        return self.image[x_slice,y_slice,z_slice]

    def from_roi(self, coords, level=0, channel=0, padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        coords = [int(coord) for coord in coords]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.roi[0:3] 
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.roi[:3],self.roi[3:])]

        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.roi[0]+xlp,x_max-self.roi[0]-xhp)
        y_slice = slice(y_min-self.roi[1]+ylp,y_max-self.roi[1]-yhp)
        z_slice = slice(z_min-self.roi[2]+zlp,z_max-self.roi[2]-zhp) 
        img = self.image[x_slice,y_slice,z_slice]

        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), padding) # padding can be constant or reflect

        return padded


def wrap_image(image_path):
    if 'ims' in image_path:
        return Ims(image_path)
    elif 'zarr.zip' in image_path:
        return ZipZarr(image_path)
    elif 'tif' in image_path:
        return Tiff(image_path)
    else:
        raise Exception("image type not supported yet") 


if __name__ == '__main__':
    pass