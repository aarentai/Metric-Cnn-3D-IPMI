import torch, sys
sys.path.append('../Packages')
from util import riemann

def eigen_composite(u):
    theta = u[3].unsqueeze(-1).unsqueeze(-1)
    
    K = torch.zeros((*u.shape[1:], 3, 3))
    kx = u[0]
    ky = u[1]
    kz = u[2]
    K[...,0,1] = -kx
    K[...,0,2] = ky
    K[...,1,0] = kx
    K[...,1,2] = -kz
    K[...,2,0] = -ky
    K[...,2,1] = kz
    
    I = torch.zeros((*u.shape[1:], 3, 3))
    I[...,0,0] = 1
    I[...,1,1] = 1
    I[...,2,2] = 1
    
    Lamda = torch.zeros((*u.shape[1:], 3, 3))
    Lamda[...,0,0] = torch.exp(u[4])
    Lamda[...,1,1] = torch.exp(u[5])
    Lamda[...,2,2] = torch.exp(u[6])
    
    R = I + torch.mul(torch.sin(theta), K) + \
                torch.mul((1-torch.cos(theta)), torch.einsum('...ij,...jk->...ik', K, K))
    psd = torch.einsum('...ij,...jk,...lk->...il', R, Lamda, R)
    
    return psd
    
def pde(u, vector_lin, mask, differential_accuracy=2):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      u:                Output of the network - six distinct entries of a symmetric matrix.     (1, (n+1)n/2, h, w, d)
      vector_lin:     Input of the network                                                    (1, n, h, w, d)

    Returns:
      Output tensor nabla vv with shape `(n, h, w, d)`.
    """
    # [h, w, d, 3, 3]
    metric_mat = eigen_composite(u)
    tensor_mat = torch.inverse(metric_mat)
    
    nabla_vv = riemann.covariant_derivative_3d(vector_lin, metric_mat, mask, differential_accuracy=differential_accuracy)
    denominator = vector_lin[0]*vector_lin[0]+vector_lin[1]*vector_lin[1]+vector_lin[2]*vector_lin[2]
    denominator += 1-mask
    sigma = (vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1]+vector_lin[2]*nabla_vv[2])/(denominator)*mask
    
    return torch.stack((nabla_vv[0]-sigma*vector_lin[0],nabla_vv[1]-sigma*vector_lin[1],nabla_vv[2]-sigma*vector_lin[2]),0)
