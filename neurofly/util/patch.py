import numpy as np

def get_patch_coords(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    grid_count = [i//block_size if i%block_size==0 else i//block_size+1 for i in volume_size]
    hist = np.zeros(grid_count, np.uint16)
    indices = np.where(hist==0)
    indices = np.array(indices).transpose()*block_size
    indices = indices[indices[:,2].argsort()]
    return indices


def get_patch_rois(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    upper_bound = [i+j for i,j in zip(origin,volume_size)]
    block_coords = get_patch_coords(roi,block_size)
    rois = []
    for coord in block_coords:
        c1 = [i+j for i,j in zip(coord,origin)]
        c2 = [i+block_size if i+block_size<j else j for i,j in zip(c1,upper_bound)]
        size = [j-i for i,j in zip(c1,c2)]
        rois.append(c1+size)
    return rois


def get_subregions(region, subregion_size, overlap):
    x0, y0, z0 = region[:3]
    w, h, d = region[3:]
    w_s, h_s, d_s = subregion_size
    o_w, o_h, o_d = overlap
    subregions = []
    step_x = w_s - o_w
    step_y = h_s - o_h
    step_z = d_s - o_d
    for x in range(x0, x0 + w, step_x):
        for y in range(y0, y0 + h, step_y):
            for z in range(z0, z0 + d, step_z):
                if x + w_s <= x0 + w and y + h_s <= y0 + h and z + d_s <= z0 + d:
                    subregions.append([x, y, z, w_s, h_s, d_s])
    return subregions


def patchify_without_splices(roi,patch_size,splices=300):
    rois = []
    xs = list(range(roi[0],roi[0]+roi[3],patch_size))
    xs.append(roi[0]+roi[3])

    ys = list(range(roi[1],roi[1]+roi[4],patch_size))
    ys.append(roi[1]+roi[4])

    if patch_size%splices==0:
        zs = [z for z in range(roi[2],roi[2]+roi[5]) if z%splices==0 or z%patch_size==0]
        if roi[2]%splices!=0:
            zs.insert(0,roi[2])
    else:
        zs = list(range(roi[2],roi[2]+roi[5],patch_size))

    zs.append(roi[2]+roi[5])

    for x1,x2 in zip(xs[:-1],xs[1:]):
        for y1,y2 in zip(ys[:-1],ys[1:]):
            for z1,z2 in zip(zs[:-1],zs[1:]):
                rois.append([x1,y1,z1,x2-x1,y2-y1,z2-z1])
    return rois
