import os
import pickle as pkl
import scipy.io as sio
import numpy as np
import cv2

# TODO
LOG_OR_NOT = 1
# TODO
BATCH_FRAME_NUM = 30
# TODO
DCT_NUM = 10
DCT_MAT_PATH = os.getcwd() + '/models/DCT_Basis/%d.mat' % BATCH_FRAME_NUM
# TODO
#tem_j2d_ids = [0, 1, 4, 5, 6, 7, 10, 11]
#tem_smpl_joint_ids = [8, 5, 4, 7, 21, 19, 18, 20]
# TODO
POSE_PRIOR_PATH = os.getcwd() + '/models/Prior/genericPrior.mat'


def load_dct_base():
	mtx = sio.loadmat(DCT_MAT_PATH, squeeze_me=True, struct_as_record=False)	
	mtx = mtx['D']
	mtx = mtx[:DCT_NUM]

	return np.array(mtx)

def load_initial_param():
	pose_prior = sio.loadmat(POSE_PRIOR_PATH, squeeze_me=True, struct_as_record=False)
	pose_mean = pose_prior['mean']
	pose_covariance = np.linalg.inv(pose_prior['covariance'])
	#zero_shape = np.ones([13]) * 1e-8 # extra 3 for zero global rotation
	#zero_trans = np.ones([3]) * 1e-8
	#initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)

	return pose_mean, pose_covariance
