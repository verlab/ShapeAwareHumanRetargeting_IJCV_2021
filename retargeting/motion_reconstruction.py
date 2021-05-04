"""
Sample usage:
python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl
#
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.motion_tools import motion_smooth_spline
from src.motion_tools import motion_smooth_conv
from src.motion_tools import motion_interpolation_outliers

from src.retargeting_tools import quaternion_interpolation

import src.config
from src.tf_smpl.batch_smpl import SMPL

import os as _os
import os.path as _path
import glob as _glob
import cv2 as _cv2

import _pickle as _pickle
import copy
import pdb
import math


flags.DEFINE_string('pose_path', '../data/box/smpl_poses/', 'Pose to run')

flags.DEFINE_string('folder_pose_suffix', '.pkl', 'suffix of pkl poses')

flags.DEFINE_integer(
    'model_type', 1,
    '(0) neutral,(1) male and (2) female')

flags.DEFINE_integer(
    'window', 60,
    'otimization window size')

flags.DEFINE_float(
    's_spline', 0.6,
    'smooth threshold')

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True


# load the source motion model (shape and pose) of every frame
def load_model_parameter(filename_source_dir):
    with open(filename_source_dir,'rb') as f:
            cam = _pickle.load(f, encoding='latin1')
    return cam

def smpl_regression(model_path):

    tf.reset_default_graph() 
    g_1 = tf.Graph()

    with g_1.as_default():
        Betas = tf.placeholder(dtype = tf.float32, shape = [1,10])
        Thetas = tf.placeholder(dtype = tf.float32, shape = [1,72])
        Trans = tf.placeholder(dtype = tf.float32, shape = [1,3])
        model = SMPL(model_path, joint_type=config.joint_type)
        Js,A_global_tf = model(Betas,Thetas,get_skin=False,get_A_global=True)
        Joints = model.J_transformed + Trans
        return g_1,Betas,Thetas,Trans,Joints,A_global_tf

def buid_graph_body(trans_0,pose_0,model_path,joints_0):
    
    tf.reset_default_graph() 
    g_2 = tf.Graph()
    

    with g_2.as_default(): 
       betas = tf.placeholder(dtype = tf.float32, shape = (trans_0.shape[0],10))  
       #pose = tf.Variable(pose_0,dtype = tf.float32)
       #pose_new = tf.concat([pose, tf.zeros((1,69))], axis=1) 
       pose_new = tf.Variable(pose_0,dtype = tf.float32)
       trans_new = tf.Variable(trans_0,dtype = tf.float32)

       pose = tf.constant(pose_0,dtype = tf.float32)
       trans = tf.constant(trans_0,dtype = tf.float32)
       joints = tf.constant(joints_0,dtype = tf.float32)
       
            
       model = SMPL(model_path, joint_type=config.joint_type)
       Js = model(betas,pose_new,get_skin=False)
       
       joints_new = model.J_transformed + tf.stack([tf.stack([trans_new[i,:] for j in range(24)]) for i in range(trans_0.shape[0])])
       
     
       loss_restriction_3d = tf.square(joints - joints_new)
     
      
       loss = tf.reduce_sum(loss_restriction_3d)

       error = tf.train.AdamOptimizer(1e-2).minimize(loss)

       return g_2,betas,pose_new,trans_new,error,loss

def otimization(pose_0,trans_0,betas,all_betas,model_path,Jtr_smooth):

    g_2,betas_in,pose_new,trans_new,error,loss =  buid_graph_body(trans_0,pose_0,model_path,Jtr_smooth)
    
    with tf.Session(config=config_tf,graph=g_2) as sess:
        sess.run(tf.global_variables_initializer())
        for s in range(300):
            pose_resposta = sess.run([pose_new,trans_new,loss,error], feed_dict = {betas_in:all_betas})
            #print (pose_resposta[2])
    sess.close()


    return pose_resposta[0],pose_resposta[1]

def motion_reconstruction_spline(motion_source,model_path,gap_ini,gap_final):  
   
    theta_motion_read = np.zeros(len(motion_source)*72)
    translation_motion = np.zeros(len(motion_source)*3)
    all_betas = np.zeros((len(motion_source),10))
      

    for i,motionfile in enumerate(motion_source):         
       shape_pose = load_model_parameter(motionfile)    
       theta_motion_read[i*72:i*72+72] = np.array(shape_pose['pose'])
       translation_motion[i*3:i*3+3] = shape_pose['trans']
       betas = np.array(shape_pose['betas'])
       all_betas[i] = np.array(shape_pose['betas'])


    theta_source = theta_motion_read    
    
    # get joint positions as a function of model pose, betas and trans, 
    A_global = [None] * len(motion_source)
    Jtr_motion = [None] * len(motion_source) 
  

    g_1,Betas,Thetas,Trans,Joints,A_global_tf =  smpl_regression(model_path)

    with tf.Session(config=config_tf,graph=g_1) as sess:
           sess.run(tf.global_variables_initializer())
           for i in range(len(motion_source)):
               pose_resposta_2 = sess.run([Joints,A_global_tf], feed_dict = {Thetas:np.reshape(theta_source[i*72:i*72 + 72],(1,-1)),Betas:np.reshape(betas,(1,-1)),Trans:np.reshape(translation_motion[i*3:i*3 +3],(1,-1))})
               Jtr_motion[i] = np.reshape(pose_resposta_2[0],(24,3))
               A_global[i] = np.reshape(pose_resposta_2[1],(24,4,4))
              
    sess.close()

    Njoints = 24
    Nframes = len(motion_source)
    smooth = 'spline'
    if (smooth == 'conv'):
        # convolution mode with Gaussian kernel of max size 31
        matrix = motion_smooth_conv(Jtr_motion)
        # all frames are considered inliers
        frame_inliers = np.ones(Nframes)

    elif(smooth == 'spline'):
        # Smoothing spline mode for the pose
        # change smoothing thrs here (pass it as parameter later?)
        smoothing = config.s_spline
        matrix = motion_smooth_spline(Jtr_motion, smoothing)
        # we have already interpolated the joint 3d locations
        # Now we need to replace the thetas and translation of the outliers frames (if any)
        # with the information of nearest inlier frames        
        frame_inliers = matrix[3]

    xjoints_sm = matrix[0]; yjoints_sm = matrix[1]; zjoints_sm = matrix[2]
    # we interpolate theta and translation vectors to interpolate information from outliers
    # check if there is outliers
    if (np.sum(frame_inliers) < frame_inliers.size):
        #pdb.set_trace()
        theta_resampled, trans_resampled = motion_interpolation_outliers(theta_source, translation_motion, frame_inliers)
        #theta_source = theta_resampled
        theta_source = (quaternion_interpolation(np.reshape(theta_source,(len(motion_source),1,-1)),frame_inliers)).ravel()
        translation_motion = trans_resampled 
        #pdb.set_trace()
    
    # Recover 3D matrices 
    pose_0 = np.zeros((len(motion_source),72))

    trans_0 = np.zeros((len(motion_source),3))

    Jtr_smooth = np.zeros((len(motion_source),Njoints,3))

    for i in range(len(motion_source)):             
        for ii in range(Njoints):
            Jtr_smooth[i,ii,0] = xjoints_sm[ii][i]
            Jtr_smooth[i,ii,1] = yjoints_sm[ii][i]
            Jtr_smooth[i,ii,2] = zjoints_sm[ii][i] 

        pose_0[i] = theta_source[i*72:i*72+72] 
        trans_0[i] = translation_motion[i*3:i*3+3]

   

    # otimization 

    pose,trans = otimization(pose_0,trans_0,betas,all_betas,model_path,Jtr_smooth)

    
    # save result 

    for j in range(len(motion_source)):
        if (j >= gap_ini) and ((len(motion_source) - gap_final) > j): 

            with open(motion_source[j], 'rb') as fout:
                print(motion_source[j])
                cam = _pickle.load(fout,encoding='latin1') 
        
            #cam['betas'] = set_betas[0,:]
            cam['pose'] = pose[j]
            cam['trans'] = trans[j]
            #cam['model_type'] = source_model_type

            with open(motion_source[j] + '_sm.pkl', 'wb') as fout2:
                _pickle.dump(cam, fout2,protocol=2)
 
      

def main(filename_source):      


    # search pose files    
    
    if _os.path.isdir(filename_source):
        folder_name2 = filename_source[:]
        motion_sourceg = sorted(_glob.glob(_os.path.join(folder_name2, '*' + config.folder_pose_suffix)))              
    else:
        print("Not found smpl pkl")
        exit()

    
    #define the correct model to load

    if config.model_type == 1:
         model_path = config.smpl_model_path_m
    elif config.model_type == 2: 
        model_path = config.smpl_model_path_f 
    else: 
        model_path = config.smpl_model_path

    window = config.window
    gap_w = int(window/3)

    #print(len(motion_sourceg))
    #print(int(math.floor(len(motion_sourceg)/(step*window))))
    #print(len(motion_sourceg[0*step*window:step:(1)*step*window]))

    for i in range(int(math.floor(len(motion_sourceg)/(window)))):
        final = np.minimum(int((i +1)*window + gap_w),len(motion_sourceg)) 
        gap_final = final - (i +1)*window 
        gap_ini = gap_w        
 
        if i == 0:
            gap_ini = 0          
               

        motion_source = motion_sourceg[i*window - gap_ini:(i+1)*window + gap_final] 
        #print (i*window - gap_ini,(i+1)*window + gap_final,gap_ini,gap_final)   
     
        motion_reconstruction_spline(motion_source,model_path,gap_ini,gap_final)
    if int(math.floor(len(motion_sourceg)/(window)))*(window) < len(motion_sourceg):
        ini_tmp = int(math.floor(len(motion_sourceg)/(window)))*window
        ini = np.maximum(0,ini_tmp - gap_w) 
        gap_ini = ini_tmp - ini        
        
        motion_source = motion_sourceg[ini:len(motion_sourceg)]
        #print (ini,len(motion_sourceg),gap_ini) 
        motion_reconstruction_spline(motion_source,model_path,gap_ini,0)

    
if __name__ == '__main__':
   config = flags.FLAGS
   config(sys.argv)
   main(config.pose_path) 
