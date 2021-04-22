
"""
Sample usage:

python retargeting.py --motion_path ../data/box/ --c_pose ./data/8-views/smpl_consensus_shape.pkl


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import quaternion

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import MuVs_util

from src.retargeting_tools import smpl_regression
from src.retargeting_tools import Restriction

from src.retargeting_tools import retargeting_otimization

import src.config

from src.tf_smpl.batch_smpl import SMPL
from scipy.ndimage import filters
from scipy import ndimage

import os as _os
import os.path as _path
import glob as _glob
import cv2 as _cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

flags.DEFINE_string('motion_path', '../data/box/', 'Motion to run')
flags.DEFINE_string(
    'c_pose', '../data/8-views/smpl_consensus_shape.pkl',
    'If specified, uses the consensus pose betas and model_type')

flags.DEFINE_string('pkl_end', 'sm.pkl', 'retargeting target pkl ends')
flags.DEFINE_string('pkl_end_out', '_ret.pkl', 'retargeting out pkl ends')

flags.DEFINE_string('sub_pose_path', 'smpl_pose', 'sub folder of motion_path with pkl files')
flags.DEFINE_string('sub_img_path', 'images', 'sub folder of motion_path with images')
flags.DEFINE_string('sub_restritions_path', 'restrictions-2D.pkl', 'relative path of restrictions')

import _pickle as _pickle


import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting
import pdb
import quaternion
import copy
import pdb

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True


def load(config,motion_path):

    if _os.path.isdir(motion_path):
        processing_folder = True
        folder_name = motion_path[:]        
        images = sorted(_glob.glob(_os.path.join(motion_path + '/' + config.sub_img_path+ '/', '*')))        
        images = [im for im in images if not im.endswith('vis.jpg')]
        pose_names =  sorted(_glob.glob(_os.path.join(motion_path + '/' + config.sub_pose_path + '/', '*' + config.pkl_end)))
        restriction_name = motion_path + "/" + config.sub_restritions_path
              
    else:
        print ("Error load motion")
        exit()
   
    #print (images) 
    #print (images[0])

    if _path.isfile(restriction_name): 
        with open(restriction_name, 'rb') as fout:
            all_restrictions = _pickle.load(fout, encoding='latin1') 
    else:
        all_restrictions = []  
   
    #weights = [[] for _ in range(25)]
    trans = [[] for _ in range(len(images))]
    joints = [[] for _ in range(len(images))]
    thetas = [[] for _ in range(len(images))]
    

    motion_betas = np.zeros(10)
    motion_model_type = 0 
    flength  = 1200.0 
    cont_motion = 0
    for image_name, pose_name in zip(images, pose_names):            
        with open(pose_name, 'rb') as fout:
           cam = _pickle.load(fout, encoding='latin1')

        #print (len(cam['pose'])) 

        trans[cont_motion].append(cam['trans'])
        thetas[cont_motion].append(cam['pose'])        
    
        if cont_motion == 0:
            if 'model_type' in cam.keys():
                motion_model_type = cam['model_type']
            flength = cam['f']
            motion_betas = cam['betas']
        
        cont_motion = cont_motion + 1               
       
    

    my_shape = _cv2.imread(images[0]).shape 
   
    thetas = np.array(thetas)
    #print (thetas.shape)

    #thetas = np.reshape(thetas,(-1,72))
    trans = np.array(trans)
    betas = np.reshape(motion_betas,(1,-1))
    

    if motion_model_type == 1:
       g_1,Betas,Thetas,Trans,Joints =  smpl_regression(config.smpl_model_path_m)
    elif motion_model_type == 2:
       g_1,Betas,Thetas,Trans,Joints =  smpl_regression(config.smpl_model_path_f)
    else:
       g_1,Betas,Thetas,Trans,Joints =  smpl_regression(config.smpl_model_path)

       
    with tf.Session(config=config_tf,graph=g_1) as sess:
           sess.run(tf.global_variables_initializer())
           for i in range(len(images)):
               pose_resposta_2 = sess.run([Joints], feed_dict = {Thetas:thetas[i],Betas:betas,Trans:trans[i]})
               joints[i].append(pose_resposta_2[0])
                 

    sess.close()
  
    joints = np.reshape(np.array(joints),(-1,24,3))
    

    #print (joints.shape)
    trans = np.reshape(trans,(-1,3))
        
    return joints,trans,my_shape,flength,motion_betas,motion_model_type,thetas,all_restrictions,pose_names 



def write(config,result_trans,result_thetas,pose_names,source_betas,source_model_type): 

    for i,pose_name in enumerate(pose_names):            
        with open(pose_name, 'rb') as fout:
           cam = _pickle.load(fout, encoding='latin1') 

        cam['betas'] = np.reshape(source_betas,cam['betas'].shape)

        cam['pose'] = np.reshape(result_thetas[i],cam['pose'].shape)
     
        cam['trans'] = np.reshape(result_trans[i],np.array(cam['trans']).shape) 

        cam['model_type'] = source_model_type        
        
        #pdb.set_trace()
        with open(pose_name + config.pkl_end_out, 'wb') as fout2:
           _pickle.dump(cam, fout2,protocol=2)




def main(config,motion_path):

    if config.c_pose == None:
        source_betas = np.zeros((1,10))
        source_model_type = 0
    else:
       with open(config.c_pose,'rb') as f:
             avatar = _pickle.load(f,encoding='latin1')
       source_betas = np.reshape(avatar['betas'][:10],(1,-1))
       source_model_type = avatar['model_type']          
   
    joints,trans,my_shape,flength,motion_betas,motion_model_type,motion_thetas,all_restrictions,pose_names = load(config,motion_path)

    
    result_trans,result_thetas = retargeting_otimization(joints,trans,my_shape,flength,motion_betas,motion_model_type,motion_thetas,all_restrictions,source_model_type,source_betas)


    write(config,result_trans,result_thetas,pose_names,source_betas,source_model_type)

      

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv) 
    main(config,config.motion_path)


