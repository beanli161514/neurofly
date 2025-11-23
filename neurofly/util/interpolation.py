import numpy as np
from scipy.interpolate import splprep, splev

def cal_frame(u, tck):
    '''
    calculate Frenet-Serret Frame
    '''
    P = np.asarray(splev(u, tck=tck)).transpose()

    # Tangent
    r_t = np.asarray(splev(u, tck, der=1)).transpose()
    T = r_t/np.linalg.norm(r_t, axis=1, keepdims=True)

    # Curvature
    r_tt = np.asarray(splev(u, tck, der=2)).transpose()
    kappa = np.linalg.norm(np.cross(r_t,r_tt),axis=1,keepdims=True) / np.linalg.norm(r_t,axis=1,keepdims=True)**3
    N = r_tt/np.linalg.norm(r_tt, axis=1, keepdims=True)
    N = N - np.sum(N * T, axis=1, keepdims=True) * T
    N = N/np.linalg.norm(N, axis=1, keepdims=True)
    C = kappa*N

    return P, T, C


def cal_tck(ctrl_p:np.ndarray, degree:int=4):
    '''
    calculate parameters 't、c、k' for spline
    '''
    x = ctrl_p[:,0]
    y = ctrl_p[:,1]
    z = ctrl_p[:,2]
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) + np.abs(np.diff(z)) > 0)
    x = np.concatenate([x[okay], x[-1:]])
    y = np.concatenate([y[okay], y[-1:]])
    z = np.concatenate([z[okay], z[-1:]])
    tck, u = splprep([x,y,z], s=0, k=degree)
    return tck, u


def FSM(ctrl_p:np.ndarray, degree:int=4, sample_num:int=20):
    '''
    Frenet–Serret Frame
    '''
    tck, u = cal_tck(ctrl_p, degree)
    new_u = np.linspace(0, 1, sample_num)
    
    P,T,C = cal_frame(u=new_u, tck=tck)

    return P,T,C