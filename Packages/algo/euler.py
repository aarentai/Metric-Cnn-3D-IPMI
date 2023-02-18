from lazy_imports import np
import math
from util.tensors import *
from util import riemann
from data import io

def get_vec_at_point_3d(x, y, z, vector_field, prev_angle):
  # return first and second component of eigenvector at a point, and associated angle
  vect = vect_interp_3d(x, y, z, vector_field)

  # important!!!!!!!!!!!!
  '''
  When using the eigenvector calculate by myself:
  Because when the principal eigenvector is almost horizontal, say at the top of the annulus,
  the eigenvector becomes extremely small, like [0.009986493070520448 1.9950060037743356e-05]
  so we have to adjust it to [1 0] manually.
  When the tensor is very vertical or horizontal, it's typically [6 0; 0 1] or [1 0; 0 6]
  '''
  u, v, w = vect[0], vect[1], vect[2]

  # important too!!!!!!!!
  angle1 = math.atan2(v, u)
  angle2 = math.atan2(-v, -u)
  if abs(angle1 - prev_angle) < abs(angle2 - prev_angle):
    # keep the sign of eigenvector
    new_angle = angle1
  else:
    u = -u
    v = -v
    w = -w
    new_angle = angle2
  return(u, v, w, new_angle)

def eulerpath_vectbase_3d(vector_lin, mask_image, start_coordinate, delta_t=0.25, iter_num=700, filename = '', both_directions=False):
  # calculating first eigenvector
#   (x, y, z) = start_coordinate
#   u, v, w = vector_lin[0,x,y,z], vector_lin[1,x,y,z], vector_lin[2,x,y,z]

  if both_directions:
    back_x, back_y, back_z = eulerpath_vectbase_3d(vector_lin, mask_image, start_coordinate, -delta_t, iter_num, filename, both_directions=False)
  print("Euler starting eigenvector:", start_coordinate)
  prev_angle = math.atan2(start_coordinate[1], start_coordinate[0])
  gamma = np.zeros((iter_num,3))
  gamma_dot = np.zeros((iter_num,3))
  gamma[0] = start_coordinate
  gamma_dot[0] = vect_interp_3d(gamma[0,0], gamma[0,1], gamma[0,2], vector_lin)

  DV = riemann.get_jacobian_3d(vector_lin, mask_image)
  dvv = np.einsum('...ij,j...->i...', DV, vector_lin)
  points_x = []
  points_y = []
  points_z = []

  # calculating following eigenvectors
  for i in range(1,iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
#     gamma_dot[i] = vect_interp_2d(gamma[i-1,0], gamma[i-1,1], vector_lin)+delta_t*vect_interp_2d(gamma[i-1,0], gamma[i-1,1], dvv)
#     gamma_dot[i] = gamma_dot[i-1]+delta_t*vect_interp_3d(gamma[i-1,0], gamma[i-1,1], gamma[i-1,2], dvv)
#     (u, v, w, prev_angle) = get_vec_at_point_3d(gamma[i-1,0], gamma[i-1,1], gamma[i-1,2], vector_lin, prev_angle)
    gamma_dot[i] = gamma_dot[i-1]+delta_t*vect_interp_3d(gamma[i-1,0], gamma[i-1,1], gamma[i-1,2], dvv)
    if np.inner(-gamma_dot[i],gamma_dot[i-1])>np.inner(gamma_dot[i],gamma_dot[i-1]):
      gamma_dot[i] = -gamma_dot[i]

    if (math.ceil(gamma[i,0]) >= 0 and math.ceil(gamma[i,0]) < np.size(vector_lin[0], 0)
        and math.ceil(gamma[i,1]) >= 0 and math.ceil(gamma[i,1]) < np.size(vector_lin[0], 1)
        and math.ceil(gamma[i,2]) >= 0 and math.ceil(gamma[i,2]) < np.size(vector_lin[0], 2)
        and mask_image[int(math.ceil(gamma[i,0])), int(math.ceil(gamma[i,1])), int(math.ceil(gamma[i,2]))] > 0):
      points_x.append(gamma[i,0])
      points_y.append(gamma[i,1])
      points_z.append(gamma[i,2])
    else:
      break

    (u, v, w, prev_angle) = get_vec_at_point_3d(gamma[i,0], gamma[i,1], gamma[i,2], vector_lin, prev_angle)

  if both_directions:
    points_x = points_x[::-1] + back_x
    points_y = points_y[::-1] + back_y
    points_z = points_z[::-1] + back_z
    
  if filename:
    io.writePath3D(points_x, points_y, points_z, filename)

  return points_x, points_y, points_z

def batch_eulerpath_vectbase_3d(vector_lin, mask_image, start_coordinate, delta_t=0.25, iter_num=700, filename = '', both_directions=False):

  if both_directions:
    back_x, back_y, back_z = batch_eulerpath_vectbase_3d(vector_lin, mask_image, start_coordinate, -delta_t, iter_num, filename, both_directions=False)

  num_paths = start_coordinate.shape[0]
  xsz, ysz, zsz = vector_lin.shape[1:4]
  prev_angle = np.zeros(num_paths)
  for p in range(num_paths):
    prev_angle[p] = math.atan2(start_coordinate[p,1], start_coordinate[p,0])
    
  gamma = np.zeros((num_paths,iter_num,3))
  gamma_dot = np.zeros((num_paths,iter_num,3))
  gamma[:,0] = start_coordinate
  gamma_dot[:,0] = batch_vect_interp_3d(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], vector_lin)

  DV = riemann.get_jacobian_3d(vector_lin, mask_image)
  dvv = np.einsum('...ij,j...->i...', DV, vector_lin)

  points_x = [np.zeros((iter_num-2)) for p in range(num_paths)]
  points_y = [np.zeros((iter_num-2)) for p in range(num_paths)]
  points_z = [np.zeros((iter_num-2)) for p in range(num_paths)]
  continue_path = np.ones((num_paths))
    
  # calculating following eigenvectors
  for i in range(1,iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
#     gamma_dot[i] = vect_interp_2d(gamma[i-1,0], gamma[i-1,1], vector_lin)+delta_t*vect_interp_2d(gamma[i-1,0], gamma[i-1,1], dvv)
#     gamma_dot[i] = gamma_dot[i-1]+delta_t*vect_interp_3d(gamma[i-1,0], gamma[i-1,1], gamma[i-1,2], dvv)
#     (u, v, w, prev_angle) = get_vec_at_point_3d(gamma[i-1,0], gamma[i-1,1], gamma[i-1,2], vector_lin, prev_angle)
    gamma_dot[:,i] = gamma_dot[:,i-1]+delta_t*batch_vect_interp_3d(gamma[:,i-1,0], gamma[:,i-1,1], gamma[:,i-1,2], dvv)
    for p in range(num_paths):
      if np.inner(-gamma_dot[p,i],gamma_dot[p,i-1])>np.inner(gamma_dot[p,i],gamma_dot[p,i-1]):
        gamma_dot[p,i] = -gamma_dot[p,i]

    for p in range(num_paths):
      if ( continue_path[p]
       and np.ceil(gamma[p, i, 0]) >= 0 and np.ceil(gamma[p, i, 0]) < xsz
       and np.ceil(gamma[p, i, 1]) >= 0 and np.ceil(gamma[p, i, 1]) < ysz
       and np.ceil(gamma[p, i, 2]) >= 0 and np.ceil(gamma[p, i, 2]) < zsz
       and (mask_image[int(np.ceil(gamma[p, i, 0])), int(np.ceil(gamma[p, i, 1])), int(np.ceil(gamma[p, i, 2]))] > 0)):
          active_path = True
          points_x[p][i-2] = gamma[p, i, 0]
          points_y[p][i-2] = gamma[p, i, 1]
          points_z[p][i-2] = gamma[p, i, 2]
      else:
        # truncate and stop
        points_x[p] = points_x[p][:i-2]
        points_y[p] = points_y[p][:i-2]
        points_z[p] = points_z[p][:i-2]
        continue_path[p] = 0
        
    if not active_path:
      break
    
  if both_directions:
    for p in range(num_paths):
      points_x[p] = np.concatenate((points_x[p][::-1], back_x[p]))
      points_y[p] = np.concatenate((points_y[p][::-1], back_y[p]))
      points_z[p] = np.concatenate((points_z[p][::-1], back_z[p]))
    
  return points_x, points_y, points_z
