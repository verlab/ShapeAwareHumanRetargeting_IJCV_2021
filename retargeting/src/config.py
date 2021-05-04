"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json

import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()

SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_MODEL_PATH_f = osp.join(model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
SMPL_MODEL_PATH_m = osp.join(model_dir, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
#SMPL_MODEL_PATH = SMPL_MODEL_PATH_m

SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')


flags.DEFINE_string('smpl_model_path_f', SMPL_MODEL_PATH_f,
                    'path to the female smpl model')
flags.DEFINE_string('smpl_model_path_m', SMPL_MODEL_PATH_m,
                    'path to the male smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')


flags.DEFINE_string(
    'joint_type', 'cocoplus',
    'cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL')



