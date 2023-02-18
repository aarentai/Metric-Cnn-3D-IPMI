from lazy_imports import np
from lazy_imports import torch
from util import diff
from data.convert import get_framework


def get_christoffel_symbol_3d(metric_mat, mask, differential_accuracy=2):
    fw, fw_name = get_framework(metric_mat)
    if fw_name=='torch':
        tensor_mat = torch.inverse(metric_mat)
    if fw_name=='numpy':
        tensor_mat = np.linalg.inv(metric_mat)
    
    go11, go12, go13, go21, go22, go23, go31, go32, go33 = metric_mat[...,0,0], metric_mat[...,0,1], metric_mat[...,0,2], metric_mat[...,1,0], metric_mat[...,1,1], metric_mat[...,1,2], metric_mat[...,2,0], metric_mat[...,2,1], metric_mat[...,2,2]
    gi11, gi12, gi13, gi21, gi22, gi23, gi31, gi32, gi33 = tensor_mat[...,0,0], tensor_mat[...,0,1], tensor_mat[...,0,2], tensor_mat[...,1,0], tensor_mat[...,1,1], tensor_mat[...,1,2], tensor_mat[...,2,0], tensor_mat[...,2,1], tensor_mat[...,2,2]
    
    d1_go11 = diff.get_first_order_derivative(go11, direction=0, accuracy=differential_accuracy)
    d2_go11 = diff.get_first_order_derivative(go11, direction=1, accuracy=differential_accuracy)
    d3_go11 = diff.get_first_order_derivative(go11, direction=2, accuracy=differential_accuracy)
    d1_go12 = diff.get_first_order_derivative(go12, direction=0, accuracy=differential_accuracy)
    d2_go12 = diff.get_first_order_derivative(go12, direction=1, accuracy=differential_accuracy)
    d3_go12 = diff.get_first_order_derivative(go12, direction=2, accuracy=differential_accuracy)
    d1_go13 = diff.get_first_order_derivative(go13, direction=0, accuracy=differential_accuracy)
    d2_go13 = diff.get_first_order_derivative(go13, direction=1, accuracy=differential_accuracy)
    d3_go13 = diff.get_first_order_derivative(go13, direction=2, accuracy=differential_accuracy)
    d1_go21 = diff.get_first_order_derivative(go21, direction=0, accuracy=differential_accuracy)
    d2_go21 = diff.get_first_order_derivative(go21, direction=1, accuracy=differential_accuracy)
    d3_go21 = diff.get_first_order_derivative(go21, direction=2, accuracy=differential_accuracy)
    d1_go22 = diff.get_first_order_derivative(go22, direction=0, accuracy=differential_accuracy)
    d2_go22 = diff.get_first_order_derivative(go22, direction=1, accuracy=differential_accuracy)
    d3_go22 = diff.get_first_order_derivative(go22, direction=2, accuracy=differential_accuracy)
    d1_go23 = diff.get_first_order_derivative(go23, direction=0, accuracy=differential_accuracy)
    d2_go23 = diff.get_first_order_derivative(go23, direction=1, accuracy=differential_accuracy)
    d3_go23 = diff.get_first_order_derivative(go23, direction=2, accuracy=differential_accuracy)
    d1_go31 = diff.get_first_order_derivative(go31, direction=0, accuracy=differential_accuracy)
    d2_go31 = diff.get_first_order_derivative(go31, direction=1, accuracy=differential_accuracy)
    d3_go31 = diff.get_first_order_derivative(go31, direction=2, accuracy=differential_accuracy)
    d1_go32 = diff.get_first_order_derivative(go32, direction=0, accuracy=differential_accuracy)
    d2_go32 = diff.get_first_order_derivative(go32, direction=1, accuracy=differential_accuracy)
    d3_go32 = diff.get_first_order_derivative(go32, direction=2, accuracy=differential_accuracy)
    d1_go33 = diff.get_first_order_derivative(go33, direction=0, accuracy=differential_accuracy)
    d2_go33 = diff.get_first_order_derivative(go33, direction=1, accuracy=differential_accuracy)
    d3_go33 = diff.get_first_order_derivative(go33, direction=2, accuracy=differential_accuracy)
    
    gamma1 = fw.zeros_like(metric_mat)
    gamma2 = fw.zeros_like(metric_mat)
    gamma3 = fw.zeros_like(metric_mat)
    gamma1[...,0,0] = (gi11*(d1_go11+d1_go11-d1_go11)+gi12*(d1_go12+d1_go12-d2_go11)+gi13*(d1_go13+d1_go13-d3_go11))*0.5
    gamma1[...,0,1] = (gi11*(d1_go21+d2_go11-d1_go12)+gi12*(d1_go22+d2_go12-d2_go12)+gi13*(d1_go23+d2_go13-d3_go12))*0.5
    gamma1[...,0,2] = (gi11*(d1_go31+d3_go11-d1_go13)+gi12*(d1_go32+d3_go12-d2_go13)+gi13*(d1_go33+d3_go13-d3_go13))*0.5
    gamma1[...,1,0] = (gi11*(d2_go11+d1_go21-d1_go21)+gi12*(d2_go12+d1_go22-d2_go21)+gi13*(d2_go13+d1_go23-d3_go21))*0.5
    gamma1[...,1,1] = (gi11*(d2_go21+d2_go21-d1_go22)+gi12*(d2_go22+d2_go22-d2_go22)+gi13*(d2_go23+d2_go23-d3_go22))*0.5
    gamma1[...,1,2] = (gi11*(d2_go31+d3_go21-d1_go23)+gi12*(d2_go32+d3_go22-d2_go23)+gi13*(d2_go33+d3_go23-d3_go23))*0.5
    gamma1[...,2,0] = (gi11*(d3_go11+d1_go31-d1_go31)+gi12*(d3_go12+d1_go32-d2_go31)+gi13*(d3_go13+d1_go33-d3_go31))*0.5
    gamma1[...,2,1] = (gi11*(d3_go21+d2_go31-d1_go32)+gi12*(d3_go22+d2_go32-d2_go32)+gi13*(d3_go23+d2_go33-d3_go32))*0.5
    gamma1[...,2,2] = (gi11*(d3_go31+d3_go31-d1_go33)+gi12*(d3_go32+d3_go32-d2_go33)+gi13*(d3_go33+d3_go33-d3_go33))*0.5
    gamma2[...,0,0] = (gi21*(d1_go11+d1_go11-d1_go11)+gi22*(d1_go12+d1_go12-d2_go11)+gi23*(d1_go13+d1_go13-d3_go11))*0.5
    gamma2[...,0,1] = (gi21*(d1_go21+d2_go11-d1_go12)+gi22*(d1_go22+d2_go12-d2_go12)+gi23*(d1_go23+d2_go13-d3_go12))*0.5
    gamma2[...,0,2] = (gi21*(d1_go31+d3_go11-d1_go13)+gi22*(d1_go32+d3_go12-d2_go13)+gi23*(d1_go33+d3_go13-d3_go13))*0.5
    gamma2[...,1,0] = (gi21*(d2_go11+d1_go21-d1_go21)+gi22*(d2_go12+d1_go22-d2_go21)+gi23*(d2_go13+d1_go23-d3_go21))*0.5
    gamma2[...,1,1] = (gi21*(d2_go21+d2_go21-d1_go22)+gi22*(d2_go22+d2_go22-d2_go22)+gi23*(d2_go23+d2_go23-d3_go22))*0.5
    gamma2[...,1,2] = (gi21*(d2_go31+d3_go21-d1_go23)+gi22*(d2_go32+d3_go22-d2_go23)+gi23*(d2_go33+d3_go23-d3_go23))*0.5
    gamma2[...,2,0] = (gi21*(d3_go11+d1_go31-d1_go31)+gi22*(d3_go12+d1_go32-d2_go31)+gi23*(d3_go13+d1_go33-d3_go31))*0.5
    gamma2[...,2,1] = (gi21*(d3_go21+d2_go31-d1_go32)+gi22*(d3_go22+d2_go32-d2_go32)+gi23*(d3_go23+d2_go33-d3_go32))*0.5
    gamma2[...,2,2] = (gi21*(d3_go31+d3_go31-d1_go33)+gi22*(d3_go32+d3_go32-d2_go33)+gi23*(d3_go33+d3_go33-d3_go33))*0.5
    gamma3[...,0,0] = (gi31*(d1_go11+d1_go11-d1_go11)+gi32*(d1_go12+d1_go12-d2_go11)+gi33*(d1_go13+d1_go13-d3_go11))*0.5
    gamma3[...,0,1] = (gi31*(d1_go21+d2_go11-d1_go12)+gi32*(d1_go22+d2_go12-d2_go12)+gi33*(d1_go23+d2_go13-d3_go12))*0.5
    gamma3[...,0,2] = (gi31*(d1_go31+d3_go11-d1_go13)+gi32*(d1_go32+d3_go12-d2_go13)+gi33*(d1_go33+d3_go13-d3_go13))*0.5
    gamma3[...,1,0] = (gi31*(d2_go11+d1_go21-d1_go21)+gi32*(d2_go12+d1_go22-d2_go21)+gi33*(d2_go13+d1_go23-d3_go21))*0.5
    gamma3[...,1,1] = (gi31*(d2_go21+d2_go21-d1_go22)+gi32*(d2_go22+d2_go22-d2_go22)+gi33*(d2_go23+d2_go23-d3_go22))*0.5
    gamma3[...,1,2] = (gi31*(d2_go31+d3_go21-d1_go23)+gi32*(d2_go32+d3_go22-d2_go23)+gi33*(d2_go33+d3_go23-d3_go23))*0.5
    gamma3[...,2,0] = (gi31*(d3_go11+d1_go31-d1_go31)+gi32*(d3_go12+d1_go32-d2_go31)+gi33*(d3_go13+d1_go33-d3_go31))*0.5
    gamma3[...,2,1] = (gi31*(d3_go21+d2_go31-d1_go32)+gi32*(d3_go22+d2_go32-d2_go32)+gi33*(d3_go23+d2_go33-d3_go32))*0.5
    gamma3[...,2,2] = (gi31*(d3_go31+d3_go31-d1_go33)+gi32*(d3_go32+d3_go32-d2_go33)+gi33*(d3_go33+d3_go33-d3_go33))*0.5
    
    return gamma1, gamma2, gamma3

def get_jacobian_3d(vector_lin, mask, differential_accuracy=2):
    fw, fw_name = get_framework(vector_lin)
    v1, v2, v3 = vector_lin[0], vector_lin[1], vector_lin[2]
    
    dv = fw.zeros((*vector_lin.shape[1:], 3, 3))
    dv[...,0,0] = diff.get_first_order_derivative(v1, direction=0, accuracy=differential_accuracy)
    dv[...,0,1] = diff.get_first_order_derivative(v1, direction=1, accuracy=differential_accuracy)
    dv[...,0,2] = diff.get_first_order_derivative(v1, direction=2, accuracy=differential_accuracy)
    dv[...,1,0] = diff.get_first_order_derivative(v2, direction=0, accuracy=differential_accuracy)
    dv[...,1,1] = diff.get_first_order_derivative(v2, direction=1, accuracy=differential_accuracy)
    dv[...,1,2] = diff.get_first_order_derivative(v2, direction=2, accuracy=differential_accuracy)
    dv[...,2,0] = diff.get_first_order_derivative(v3, direction=0, accuracy=differential_accuracy)
    dv[...,2,1] = diff.get_first_order_derivative(v3, direction=1, accuracy=differential_accuracy)
    dv[...,2,2] = diff.get_first_order_derivative(v3, direction=2, accuracy=differential_accuracy)
        
    return dv

def covariant_derivative_3d(vector_lin, metric_mat, mask, differential_accuracy=2):
    """
    Calculate covariant derivative w.r.t vector_lin and metric_mat
    Args:
        vector_lin, torch.Tensor, of shape [3, h, w, d]
        metric_mat, torch.Tensor, of shape [h, w, d, 3, 3]
    Returns: 
        nabla_vv, torch.Tensor
    """
    assert vector_lin.shape[0]+1==len(vector_lin.shape), 'vector_lin should follow shape of [3, h, w, d]'
    assert metric_mat.shape[-1]+2==len(metric_mat.shape), 'metric_mat should follow shape of [h, w, d, 3, 3]'
    fw, fw_name = get_framework(vector_lin)
    v1 = vector_lin[0]
    v2 = vector_lin[1]
    v3 = vector_lin[2]
    
    dvv = fw.zeros_like(vector_lin)
    dv = get_jacobian_3d(vector_lin, mask, differential_accuracy)
    dvv = fw.einsum('...ij,j...->i...', dv, vector_lin)
    
    vgammav = fw.zeros_like(vector_lin)
    Gamma1, Gamma2, Gamma3 = get_christoffel_symbol_3d(metric_mat, mask, differential_accuracy)
    '''Unfortunately, the more concise einsum is slower than the explicit expression when the field is large, and there's a little numerical difference between the two formulation.'''
#     vgammav[0] = fw.einsum('i...,i...->...', vector_lin, fw.einsum('...ij,j...->i...', Gamma1, vector_lin))
#     vgammav[1] = fw.einsum('i...,i...->...', vector_lin, fw.einsum('...ij,j...->i...', Gamma2, vector_lin))
#     vgammav[2] = fw.einsum('i...,i...->...', vector_lin, fw.einsum('...ij,j...->i...', Gamma3, vector_lin))
    vgammav[0] = Gamma1[...,0,0]*v1*v1+Gamma1[...,0,1]*v1*v2+Gamma1[...,0,2]*v1*v3+Gamma1[...,1,0]*v2*v1+Gamma1[...,1,1]*v2*v2+Gamma1[...,1,2]*v2*v3+Gamma1[...,2,0]*v3*v1+Gamma1[...,2,1]*v3*v2+Gamma1[...,2,2]*v3*v3
    vgammav[1] = Gamma2[...,0,0]*v1*v1+Gamma2[...,0,1]*v1*v2+Gamma2[...,0,2]*v1*v3+Gamma2[...,1,0]*v2*v1+Gamma2[...,1,1]*v2*v2+Gamma2[...,1,2]*v2*v3+Gamma2[...,2,0]*v3*v1+Gamma2[...,2,1]*v3*v2+Gamma2[...,2,2]*v3*v3
    vgammav[2] = Gamma3[...,0,0]*v1*v1+Gamma3[...,0,1]*v1*v2+Gamma3[...,0,2]*v1*v3+Gamma3[...,1,0]*v2*v1+Gamma3[...,1,1]*v2*v2+Gamma3[...,1,2]*v2*v3+Gamma3[...,2,0]*v3*v1+Gamma3[...,2,1]*v3*v2+Gamma3[...,2,2]*v3*v3
    
    nabla_vv = dvv + vgammav
    
    return nabla_vv
