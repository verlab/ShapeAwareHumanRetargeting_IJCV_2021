#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import io
import pickle as pickle
#from subprocess import call
import numpy as np
from csaps import csaps
import quaternion
from scipy.ndimage import filters
from scipy import ndimage
import pdb
from statsmodels.robust import mad

'''def save_to_memory(matrix):
    with io.open('motion_sm.pkl',mode='wb') as f:
        pickle.dump(matrix, f, protocol=2)

def load_from_memory():
    with io.open('motion.pkl',mode='rb') as f:
        matrix = pickle.load(f,encoding='latin1')
    return matrix'''

# motion reconstruction with smoothing spline 
# using implementation done at https://csaps.readthedocs.io/en/latest/manual.html

def motion_smooth_spline(Jtr_motion, smoothing):

    # Set outliers bounds to get outlier joints that are far from median standard dev
    std_bounds = 2
    # let's accept a number max of 30% of outliers -- i.e. max of 8 joints
    rate_outliers = 0.3
    Njoints = 24
    k_mad = 1.48 # convertion constant of robust mad to a Gaussian std without outliers
    Nframes = len(Jtr_motion)

    # smooth joint 3D trajectories
    xjoints = [None] * Njoints
    yjoints = [None] * Njoints
    zjoints = [None] * Njoints
    xjoints_sm = [None] * Njoints
    yjoints_sm = [None] * Njoints
    zjoints_sm = [None] * Njoints
    time = np.arange(Nframes)    
    
    error_pred = [None] * Njoints
    # first run per joint before outliers removal
    for ii in range(Njoints):
        xjoints[ii] = np.hstack([Jtr_motion[jj][ii,0] for jj in range(Nframes)])
        yjoints[ii] = np.hstack([Jtr_motion[jj][ii,1] for jj in range(Nframes)])
        zjoints[ii] = np.hstack([Jtr_motion[jj][ii,2] for jj in range(Nframes)])
        poses = [xjoints[ii], yjoints[ii], zjoints[ii]]
        poses_sm = csaps(time, poses, time, smooth=smoothing)
        # make use the norm here
        error = np.sum(np.absolute(poses_sm - poses), axis = 0)
        error_pred[ii] = np.where(np.isnan(error), np.Inf, error)


    # voting scheme using robust median and MAD per joint
    outliers = [None] * Njoints
    outliers_cumul = np.zeros(len(time))
    for ii in range(Njoints):
        mediane = np.median(error_pred[ii])
        # std = k*mad -- k = 1.148
        made = mad(error_pred[ii])
        # test if values are afar of 2*std using robust mad
        outliers[ii] = (np.absolute(error_pred[ii] - mediane) > std_bounds*k_mad*made)
        #pdb.set_trace()
        outliers_cumul += outliers[ii].astype(int)
        
    xjoints = [None] * Njoints
    yjoints = [None] * Njoints
    zjoints = [None] * Njoints
    ## let's accept a number max of 30% of outliers -- i.e. max of 8 joints
    max_outliers = Njoints*rate_outliers
    inlier_poses = (outliers_cumul < max_outliers)
    frame_inliers = time[inlier_poses]
    # spline with inliers per joint after outliers removal
    for ii in range(Njoints):
        xjoints[ii] = np.hstack([Jtr_motion[jj][ii,0] for jj in frame_inliers])
        yjoints[ii] = np.hstack([Jtr_motion[jj][ii,1] for jj in frame_inliers])
        zjoints[ii] = np.hstack([Jtr_motion[jj][ii,2] for jj in frame_inliers])
        poses = [xjoints[ii], yjoints[ii], zjoints[ii]]
        poses_sm = csaps(frame_inliers, poses, time, smooth=smoothing)
        xjoints_sm[ii] = poses_sm[0, :]
        yjoints_sm[ii] = poses_sm[1, :]
        zjoints_sm[ii] = poses_sm[2, :]

    return [xjoints_sm, yjoints_sm, zjoints_sm, inlier_poses]


# function to fill the Theta and translation poses of outlier frames
# this is required to help the inverve kinematics to converge in cases of real outliers
def motion_interpolation_outliers(theta, translation, inlier_poses, smoothing = 0.8):
    # Recover 3D matrices 
    Njoints = 24
    Nframes = inlier_poses.size
    poses_spline = [None] * (3*Njoints)
    trans_spline = [None] * 3

    # recover the inlier frames
    time = np.arange(Nframes)
    frame_inliers = time[inlier_poses]
    frame_outliers = time[inlier_poses == 0]
    pose_0 = np.zeros((frame_inliers.size,72))
    trans_0 = np.zeros((frame_inliers.size,3))

    # only use the frames that are inliers
    for i, j in enumerate(frame_inliers):        
        pose_0[i] = theta[j*72:j*72+72]
        trans_0[i] = translation[j*3:j*3+3]

    # fill list of thetas componenets to 1d interpolation
    for i in range(len(poses_spline)):           
        poses_spline[i] = pose_0[:,i]
    theta_sm = csaps(frame_inliers, poses_spline, frame_outliers, smooth=smoothing)

    # fill list of 3d translations to interpolation 
    for i in range(3):
        trans_spline[i] = trans_0[:,i]
    trans_sm = csaps(frame_inliers, trans_spline, frame_outliers, smooth=smoothing)
    debug = False
    if debug:
        ## for testing
        theta_backup = np.array(theta); trans_backup = np.array(translation)    
    # replace outlier motions by the interpolated ones
    for i, j in enumerate(frame_outliers):
        theta[j*72:j*72+72] = theta_sm[:,i]
        translation[j*3:j*3+3] = trans_sm[:,i]
    #pdb.set_trace()
    return theta, translation

# motion smoothing with a convolution (max window size of 31 frames) with
# Gaussian and std = 2
def motion_smooth_conv(Jtr_motion):
    # smooth rotations using quaternions but needs to include in the inverse kinematics
    ##R_global_sm = smooth_rotation_endeffectors(A_global)

    Nframes = len(Jtr_motion)
    Njoints = 24

    NN = np.minimum(31,Nframes)
    sigma = 2

    mask = np.array([0.0]*NN)
    mask[int((NN-1)/2)] = 1.0
    weights = filters.gaussian_filter1d(mask,sigma)

    # smooth joint 3D trajectories
    xjoints = [None] * Njoints
    yjoints = [None] * Njoints
    zjoints = [None] * Njoints
    xjoints_sm = [None] * Njoints
    yjoints_sm = [None] * Njoints
    zjoints_sm = [None] * Njoints

    for ii in range(Njoints):
       xjoints[ii] = np.hstack([Jtr_motion[jj][ii,0] for jj in range(Nframes)])
       yjoints[ii] = np.hstack([Jtr_motion[jj][ii,1] for jj in range(Nframes)])
       zjoints[ii] = np.hstack([Jtr_motion[jj][ii,2] for jj in range(Nframes)])
       xjoints_sm[ii] = np.hstack(ndimage.convolve(xjoints[ii], weights, mode='nearest'))
       yjoints_sm[ii] = np.hstack(ndimage.convolve(yjoints[ii], weights, mode='nearest'))
       zjoints_sm[ii] = np.hstack(ndimage.convolve(zjoints[ii], weights, mode='nearest'))

    return [xjoints_sm, yjoints_sm, zjoints_sm]


def smooth_rotation_endeffectors(list_Aglobal):

    Nframes = len(list_Aglobal)
    Njoints = 24
    NN = np.minimum(31,Nframes)
    sigma = 2
    # weights to filter angles
    mask = np.array([0.0]*NN)

    #pdb.set_trace()

    mask[int((NN-1)/2)] = 1.0
    weights = filters.gaussian_filter1d(mask,sigma)
    # read rotation and represent in quaternion form
    quater = [None] * Nframes 
    for i in range(Nframes):
        quater[i] = np.vstack((quaternion.from_rotation_matrix((g[:3, :3]))) for g in list_Aglobal[i])
    quater = quaternion.as_float_array(quater)
    
    # check consitency of the orientation of the quaterniorns (-q and q are the same rotation)
    for ii in range(Nframes):
        for jj in range(Njoints):
            if (np.dot(quater[ii,jj,0,:],quater[0,0,0,:])<0):               
               quater[ii,jj,0,:] = -quater[ii,jj,0,:]       


    # check consitency of the orientation of the quaterniorns (-q and q are the same rotation)
    for ii in range(Nframes):
        for jj in range(Njoints):
            if (np.dot(quater[ii,jj,0,:],quater[0,0,0,:])<0):
               pdb.set_trace()             
    
    #plot_orientation_sphere(quater,15)
    # smooth quaternions
    quater_smooth = np.zeros(quater.shape)        
    for i in range(Njoints):
       quatera = ndimage.convolve(quater[:,i,0,0], weights, mode='nearest')
       quaterx = ndimage.convolve(quater[:,i,0,1], weights, mode='nearest')
       quatery = ndimage.convolve(quater[:,i,0,2], weights, mode='nearest')
       quaterz = ndimage.convolve(quater[:,i,0,3], weights, mode='nearest')
       quater_smooth[:,i,0,0] = quatera
       quater_smooth[:,i,0,1] = quaterx
       quater_smooth[:,i,0,2] = quatery
       quater_smooth[:,i,0,3] = quaterz
    

    # normalize quaternions to unit sphere    
    for ii in range(Nframes):
       for jj in range(Njoints):
          #if(_np.sqrt((_np.sum(quater_smooth[ii,jj,0,:]**2)))<0.1):
          #   pdb.set_trace()
          quater_smooth[ii,jj,0,:] = quater_smooth[ii,jj,0,:]/np.sqrt((np.sum(quater_smooth[ii,jj,0,:]**2)))

    #plot_orientation_sphere(quater_smooth,15)
    #pdb.set_trace()
    # transform back smoothed orientations in matrix form    
    A_global_smooth = [None] * Nframes
    for ii in range(Nframes):
        A_global_smooth[ii] = np.vstack((quaternion.as_rotation_matrix(quaternion.as_quat_array(quater_smooth[ii,jj,0,:]))) for jj in range(Njoints))

    return A_global_smooth

if __name__ == '__main__':
    
    print("load smooth functions ok")
    #poses = load_from_memory()    
    #poses_sm = motion_smooth_spline(poses)
    #save_to_memory(poses_sm)


