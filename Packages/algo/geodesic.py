import math
from util.tensors import *
from lazy_imports import np
from lazy_imports import torch
from util import diff, maskops, riemann, tensors
from data import io

from numba import jit
# uncomment this for legit @profile when not using kernprof
def profile(blah):                
  return blah

def angle_changed(vector1, vector2):
    return np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))/np.pi*180

def get_gamma_ddot_at_point(x, y, Gamma_field, gamma_dot):
  tens = tens_interp(x,y,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])

  return(gamma_ddot)

def get_gamma_ddot_at_point_torch(x, y, Gamma_field, gamma_dot):
  tens = tens_interp_torch(x,y,Gamma_field).clone()
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])
  #gamma_ddot = -term1 - term2 - term3 - term4

  return(gamma_ddot)

def get_gamma_ddot_at_point_3d(x, y, z, Gamma_field, gamma_dot):
  tens = tens_interp_3d(x,y,z,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma13 = tens[0,2]
  Gamma22 = tens[1,1]
  Gamma23 = tens[1,2]
  Gamma33 = tens[2,2]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma13*gamma_dot[0]*gamma_dot[2]
                 +Gamma13*gamma_dot[2]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1]
                 +Gamma23*gamma_dot[1]*gamma_dot[2]
                 +Gamma23*gamma_dot[2]*gamma_dot[1]
                 +Gamma33*gamma_dot[2]*gamma_dot[2])

  return(gamma_ddot)

def get_gamma_ddot_at_point_3d_torch(x, y, z, Gamma_field, gamma_dot):
  tens = tens_interp_3d_torch(x,y,z,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma13 = tens[0,2]
  Gamma22 = tens[1,1]
  Gamma23 = tens[1,2]
  Gamma33 = tens[2,2]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma13*gamma_dot[0]*gamma_dot[2]
                 +Gamma13*gamma_dot[2]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1]
                 +Gamma23*gamma_dot[1]*gamma_dot[2]
                 +Gamma23*gamma_dot[2]*gamma_dot[1]
                 +Gamma33*gamma_dot[2]*gamma_dot[2])

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_3d(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_3d_torch(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d_torch(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

#@jit(nopython=True)
def compute_gammas_3d(tensor_field, mask_image):
  # Compute the Christoffel symbols Gamma1, Gamma2, Gamma3
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  # Compute inverse of g
  eps_11 = eps22 * eps33 - eps23 * eps23
  eps_12 = eps13 * eps23 - eps12 * eps33
  eps_13 = eps12 * eps23 - eps13 * eps22
  eps_22 = eps11 * eps33 - eps13 * eps13
  eps_23 = eps13 * eps12 - eps11 * eps23
  eps_33 = eps11 * eps22 - eps12 * eps12
  det_eps = eps11 * eps_11 + eps12 * eps_12 + eps13 * eps_13
  eps_11 = eps_11 / det_eps
  eps_12 = eps_12 / det_eps
  eps_13 = eps_13 / det_eps
  eps_22 = eps_22 / det_eps
  eps_23 = eps_23 / det_eps
  eps_33 = eps_33 / det_eps
    
  bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_3d(mask_image, False)
  d1_eps_11, d2_eps_11, d3_eps_11 = diff.gradient_bdry_3d(eps_11, bdry_idx, bdry_map)
  d1_eps_12, d2_eps_12, d3_eps_12 = diff.gradient_bdry_3d(eps_12, bdry_idx, bdry_map)
  d1_eps_13, d2_eps_13, d3_eps_13 = diff.gradient_bdry_3d(eps_13, bdry_idx, bdry_map)
  d1_eps_22, d2_eps_22, d3_eps_22 = diff.gradient_bdry_3d(eps_22, bdry_idx, bdry_map)
  d1_eps_23, d2_eps_23, d3_eps_23 = diff.gradient_bdry_3d(eps_23, bdry_idx, bdry_map)
  d1_eps_33, d2_eps_33, d3_eps_33 = diff.gradient_bdry_3d(eps_33, bdry_idx, bdry_map)
    
  Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11) + eps13 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22 + eps13 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma1_13 = (eps11 * d3_eps_11 + eps12 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps13 * (d1_eps_33)) / 2
  Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22 + eps13 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma1_23 = (eps11 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps12 * d3_eps_22 + eps13 * (d2_eps_33)) / 2
  Gamma1_33 = (eps11 * (2 * d3_eps_13 - d1_eps_33) + eps12 * (2 * d3_eps_23 - d2_eps_33) + eps13 * d3_eps_33) / 2
  Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_13, Gamma1_22, Gamma1_23, Gamma1_33))
  Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11) + eps23 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22 + eps23 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma2_13 = (eps12 * d3_eps_11 + eps22 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps23 * (d1_eps_33)) / 2
  Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22 + eps23 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma2_23 = (eps12 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps22 * d3_eps_22 + eps23 * (d2_eps_33)) / 2
  Gamma2_33 = (eps12 * (2 * d3_eps_13 - d1_eps_33) + eps22 * (2 * d3_eps_23 - d2_eps_33) + eps23 * d3_eps_33) / 2
  Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_13, Gamma2_22, Gamma2_23, Gamma2_33))
  Gamma3_11 = (eps13 * d1_eps_11 + eps23 * (2 * d1_eps_12 - d2_eps_11) + eps33 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma3_12 = (eps13 * d2_eps_11 + eps23 * d1_eps_22 + eps33 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma3_13 = (eps13 * d3_eps_11 + eps23 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps33 * (d1_eps_33)) / 2
  Gamma3_22 = (eps13 * (2 * d2_eps_12 - d1_eps_22) + eps23 * d2_eps_22 + eps33 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma3_23 = (eps13 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps23 * d3_eps_22 + eps33 * (d2_eps_33)) / 2
  Gamma3_33 = (eps13 * (2 * d3_eps_13 - d1_eps_33) + eps23 * (2 * d3_eps_23 - d2_eps_33) + eps33 * d3_eps_33) / 2
  Gamma3 = np.stack((Gamma3_11, Gamma3_12, Gamma3_13, Gamma3_22, Gamma3_23, Gamma3_33))
  return(Gamma1, Gamma2, Gamma3)
# end compute_gammas_3d

def geodesicpath_3d(tensor_lin, vector_lin, mask_image, start_coordinate, initial_velocity, delta_t=0.15, iter_num=18000, stop_angle=30, filename = '', both_directions=False):
  # Compute 3d geodesic path
  # Assumes that mask_image is already a differentiable mask
  geodesicpath_points_x = np.zeros((iter_num-2))
  geodesicpath_points_y = np.zeros((iter_num-2))
  geodesicpath_points_z = np.zeros((iter_num-2))

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction_3d(start_coordinate, tensor_field)

  if both_directions:
    back_x, back_y, back_z = geodesicpath_3d(tensor_lin, vector_lin, mask_image, start_coordinate,
                                             -init_v, delta_t, iter_num, stop_angle, filename, both_directions=False)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  tensor_mat = lin2mat(tensor_lin)
  metric_mat = np.linalg.inv(tensor_mat)
  Gamma1, Gamma2, Gamma3 = riemann.get_christoffel_symbol_3d(metric_mat, mask_image)
  nabla_vv = riemann.covariant_derivative_3d(vector_lin, metric_mat, mask_image)
  sigma = ((vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1]+vector_lin[2]*nabla_vv[2])/(vector_lin[0]**2+vector_lin[1]**2+vector_lin[2]**2+1e-2))
  sigmav = np.zeros_like(vector_lin)
  sigmav[0] = sigma*vector_lin[0]
  sigmav[1] = sigma*vector_lin[1]
  sigmav[2] = sigma*vector_lin[2]

  gamma = np.zeros((iter_num,3))
  gamma_dot = np.zeros((iter_num,3))
  gamma_ddot = np.zeros((iter_num,3))
  gamma[0] = start_coordinate
  gamma_dot[0] = init_v
#   gamma_ddot[0, 0] = -np.einsum('i,i->', gamma_dot[0], np.einsum('ij,j->i',tens_interp_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma1),gamma_dot[0]))
#   gamma_ddot[0, 1] = -np.einsum('i,i->', gamma_dot[0], np.einsum('ij,j->i',tens_interp_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma2),gamma_dot[0]))
#   gamma_ddot[0, 2] = -np.einsum('i,i->', gamma_dot[0], np.einsum('ij,j->i',tens_interp_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma3),gamma_dot[0]))
  gamma[1] = gamma[0] +delta_t*gamma_dot[0]
  
  for i in range(2,iter_num):
    Gamma1_gamma = tens_interp_3d(gamma[i-2,0], gamma[i-2,1], gamma[i-2,2], Gamma1)
    Gamma2_gamma = tens_interp_3d(gamma[i-2,0], gamma[i-2,1], gamma[i-2,2], Gamma2)
    Gamma3_gamma = tens_interp_3d(gamma[i-2,0], gamma[i-2,1], gamma[i-2,2], Gamma3)
    gamma_ddot[i-2,0] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma1_gamma,gamma_dot[i-2]))
    gamma_ddot[i-2,1] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma2_gamma,gamma_dot[i-2]))
    gamma_ddot[i-2,2] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma3_gamma,gamma_dot[i-2]))
    gamma_ddot[i-2] += vect_interp_3d(gamma[i-2,0], gamma[i-2,1], gamma[i-2,2], sigmav)
    gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
    gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
#     print(np.linalg.norm(gamma_dot[i-2]),np.linalg.norm(gamma_dot[i-1]))
    
    if (math.ceil(gamma[i, 0]) >= 0 and math.ceil(gamma[i, 0]) < vector_lin.shape[1]
       and math.ceil(gamma[i, 1]) >= 0 and math.ceil(gamma[i, 1])  < vector_lin.shape[2]
       and math.ceil(gamma[i, 2]) >= 0 and math.ceil(gamma[i, 2])  < vector_lin.shape[3]
        and (mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1])), int(math.ceil(gamma[i, 2]))] > 0)#:
       and (i==2 or angle_changed(gamma_dot[i-1], gamma_dot[i-2])<stop_angle)):
      geodesicpath_points_x[i-2] = gamma[i, 0]
      geodesicpath_points_y[i-2] = gamma[i, 1]
      geodesicpath_points_z[i-2] = gamma[i, 2]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-2]
      geodesicpath_points_y = geodesicpath_points_y[:i-2]
      geodesicpath_points_z = geodesicpath_points_z[:i-2]
      break

  if both_directions:
    geodesicpath_points_x = np.concatenate((geodesicpath_points_x[::-1], back_x))
    geodesicpath_points_y = np.concatenate((geodesicpath_points_y[::-1], back_y))
    geodesicpath_points_z = np.concatenate((geodesicpath_points_z[::-1], back_z))
    
  if filename:
    io.writePath3D(geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z, filename)

  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z

def batch_geodesicpath_3d(tensor_field, mask_image, start_coordinates, initial_velocities, delta_t=0.15, iter_num=18000, stop_angle=30, both_directions=False, Gamma1=None, Gamma2=None, Gamma3=None, sigmav=None):
  # Compute 3d geodesic path of metric, where the input tensor_field is the inverse of the metric
  # Assumes that mask_image is already a differentiable mask
  # To speed up when calling multiple times, precompute gammas by calling
  #  Gamma1, Gamma2, Gamma3 = compute_gammas_3d(tensor_field, mask_image)
  # and pass in to method

  num_paths = start_coordinates.shape[0]
  xsz, ysz, zsz = tensor_field.shape[1:4]
  tensor_mat = tensors.lin2mat(tensor_field)
  metric_mat = np.linalg.inv(tensor_mat)
  
  geodesicpath_points_x = [np.zeros((iter_num-2)) for p in range(num_paths)]
  geodesicpath_points_y = [np.zeros((iter_num-2)) for p in range(num_paths)]
  geodesicpath_points_z = [np.zeros((iter_num-2)) for p in range(num_paths)]
  continue_path = np.ones((num_paths))

  init_v = initial_velocities

  if both_directions:
    back_x, back_y, back_z = batch_geodesicpath_3d(tensor_field, mask_image, start_coordinates,
                                                   -init_v, delta_t, iter_num, stop_angle, both_directions=False,
                                                   Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, sigmav=sigmav)

  gamma = np.zeros((num_paths,iter_num,3))
  gamma_dot = np.zeros((num_paths,iter_num,3))
  gamma_ddot = np.zeros((num_paths,iter_num,3))
  gamma[:, 0, :] = start_coordinates
  gamma_dot[:, 0, :] = init_v

  gamma[:,1] = gamma[:,0] +delta_t*gamma_dot[:,0]
  
  for i in range(2,iter_num):
    Gamma1_gamma = batch_tens_interp_3d(gamma[:,i-2,0], gamma[:,i-2,1], gamma[:,i-2,2], Gamma1)
    Gamma2_gamma = batch_tens_interp_3d(gamma[:,i-2,0], gamma[:,i-2,1], gamma[:,i-2,2], Gamma2)
    Gamma3_gamma = batch_tens_interp_3d(gamma[:,i-2,0], gamma[:,i-2,1], gamma[:,i-2,2], Gamma3)
#     gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma1_gamma,gamma_dot[:,i-2]))
#     gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma2_gamma,gamma_dot[:,i-2]))
#     gamma_ddot[:,i-2,2] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma3_gamma,gamma_dot[:,i-2]))
    gamma_ddot[:,i-2,0] = -(gamma_dot[:,i-2,0]*Gamma1_gamma[:,0,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,0]*Gamma1_gamma[:,0,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,0]*Gamma1_gamma[:,0,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,1]*Gamma1_gamma[:,1,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,1]*Gamma1_gamma[:,1,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,1]*Gamma1_gamma[:,1,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,2]*Gamma1_gamma[:,2,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,2]*Gamma1_gamma[:,2,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,2]*Gamma1_gamma[:,2,2]*gamma_dot[:,i-2,2])
    gamma_ddot[:,i-2,1] = -(gamma_dot[:,i-2,0]*Gamma2_gamma[:,0,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,0]*Gamma2_gamma[:,0,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,0]*Gamma2_gamma[:,0,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,1]*Gamma2_gamma[:,1,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,1]*Gamma2_gamma[:,1,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,1]*Gamma2_gamma[:,1,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,2]*Gamma2_gamma[:,2,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,2]*Gamma2_gamma[:,2,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,2]*Gamma2_gamma[:,2,2]*gamma_dot[:,i-2,2])
    gamma_ddot[:,i-2,2] = -(gamma_dot[:,i-2,0]*Gamma3_gamma[:,0,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,0]*Gamma3_gamma[:,0,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,0]*Gamma3_gamma[:,0,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,1]*Gamma3_gamma[:,1,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,1]*Gamma3_gamma[:,1,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,1]*Gamma3_gamma[:,1,2]*gamma_dot[:,i-2,2]+\
                            gamma_dot[:,i-2,2]*Gamma3_gamma[:,2,0]*gamma_dot[:,i-2,0]+\
                            gamma_dot[:,i-2,2]*Gamma3_gamma[:,2,1]*gamma_dot[:,i-2,1]+\
                            gamma_dot[:,i-2,2]*Gamma3_gamma[:,2,2]*gamma_dot[:,i-2,2])
    if type(sigmav)==np.ndarray:
      gamma_ddot[:,i-2] += batch_vect_interp_3d(gamma[:,i-2,0], gamma[:,i-2,1], gamma[:,i-2,2], sigmav)
    gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
    gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
#     print(gamma[:,i].shape)

    active_path=False
    for p in range(num_paths):
      if ( continue_path[p]
       and np.ceil(gamma[p, i, 0]) >= 0 and np.ceil(gamma[p, i, 0]) < xsz
       and np.ceil(gamma[p, i, 1]) >= 0 and np.ceil(gamma[p, i, 1]) < ysz
       and np.ceil(gamma[p, i, 2]) >= 0 and np.ceil(gamma[p, i, 2]) < zsz
       and (mask_image[int(np.ceil(gamma[p, i, 0])), int(np.ceil(gamma[p, i, 1])), int(np.ceil(gamma[p, i, 2]))] > 0)):
#        and (i==2 or angle_changed(gamma_dot[p, i-1], gamma_dot[p, i-2])<stop_angle)):
          active_path = True
          geodesicpath_points_x[p][i-2] = gamma[p, i, 0]
          geodesicpath_points_y[p][i-2] = gamma[p, i, 1]
          geodesicpath_points_z[p][i-2] = gamma[p, i, 2]
      else:
        # truncate and stop
        geodesicpath_points_x[p] = geodesicpath_points_x[p][:i-2]
        geodesicpath_points_y[p] = geodesicpath_points_y[p][:i-2]
        geodesicpath_points_z[p] = geodesicpath_points_z[p][:i-2]
        continue_path[p] = 0
    if not active_path:
      break
  # End for each time point i

  if both_directions:
    for p in range(num_paths):
      geodesicpath_points_x[p] = np.concatenate((geodesicpath_points_x[p][::-1], back_x[p]))
      geodesicpath_points_y[p] = np.concatenate((geodesicpath_points_y[p][::-1], back_y[p]))
      geodesicpath_points_z[p] = np.concatenate((geodesicpath_points_z[p][::-1], back_z[p]))
    
  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z
# end batch_geodesicpath_3d

def geodesic_between_points_torch(tensor_field, mask_image, start_coordinate, end_coordinate, init_velocity=None, step_size=0.0001, num_iters=18000, filename = ''):
  # assumes tensor_field and mask_image are np arrays, converts to torch here
  torch_field = torch.from_numpy(tensor_field)
  mask = torch.from_numpy(mask_image)
  start_coords = torch.tensor(start_coordinate)
  end_coords = torch.tensor(end_coordinate)

  # TODO Is there a way to use pytorch batching to compute many geodesics at once?
  energy = torch.zeros((num_iters))
  init_v = torch.zeros((num_iters, 2), requires_grad=True)
  
  if init_velocity is None:
    init_v[0] = direction_torch(start_coordinate, tensor_field)
  else:
    init_v[0] = torch.tensor(init_velocity)

  all_points_x = []
  all_points_y = []
  
  for it in range(0,num_iters-1):
    end_point, points_x, points_y = geodesicpath_torch(torch_field, mask, start_coords, init_v[it], delta_t=0.15, iter_num=18000, filename = '')
    all_points_x.append(points_x)
    all_points_y.append(points_y)
    energy[it] = ((end_point[0] - end_coords[0])**2 + (end_point[1] - end_coords[1])**2)
    energy.backward()
    init_v[it+1] = init_v[it] - step_size * init_v.grad

  return(all_points_x, all_points_y, init_v, energy)
