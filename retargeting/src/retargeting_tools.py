import numpy as np
import quaternion
import tensorflow as tf
from .tf_smpl.batch_smpl import SMPL
from absl import flags
import cv2 as _cv2

from .util import renderer as vis_util
from .util import MuVs_util

import pdb

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

from csaps import csaps

class Restriction(object):
    def __init__(self,
                 my_type,
                 start,
                 end,
                 joint,
                 x= None,
                 y= None,
                 restriction_point = None,
                 free_joints= None):
        self.my_type = my_type
        self.start = start
        if restriction_point is None:
            self.restriction_point = []
        else:
            self.restriction_point = restriction_point
        self.end = end
        self.joint = joint
        self.free_joints = free_joints
        
        if (x is None) or (y is None): 
            self.x = []
            self.y = []
        else:
            self.x = x
            self.y = y


    def reshape_restriction(self,new_start,new_end):
        self.start = np.maximum(new_start,self.start) - new_start
        self.end = np.minimum(new_end,self.end) - new_start 



def smpl_regression(model_path):

    config = flags.FLAGS
    tf.reset_default_graph() 
    g_1 = tf.Graph()
    with g_1.as_default():
        Betas = tf.placeholder(dtype = tf.float32, shape = [1,10])
        Thetas = tf.placeholder(dtype = tf.float32, shape = [1,72])
        Trans = tf.placeholder(dtype = tf.float32, shape = [1,3])
        model = SMPL(model_path, joint_type=config.joint_type)
        Js = model(Betas,Thetas,get_skin=False)
        Joints = model.J_transformed + Trans
        return g_1,Betas,Thetas,Trans,Joints


def dot_quaternion(q1,q2):
    return (q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z) 


def quaternion_interpolation(motion_thetas,to_use,joints_number=24):
    
    motion_quaternions = [[] for i in range(joints_number)]  

    t_in = []
  
    for i in range(motion_thetas.shape[0]):
        if to_use[i] == True:
            for j in range(joints_number):
               motion_quaternions[j].append(quaternion.from_rotation_matrix(_cv2.Rodrigues(motion_thetas[i,0,3*j:3*j+3])[0]))
            t_in.append(i)
                

    motion_quaternions = np.array(motion_quaternions)

    for j in range(motion_quaternions.shape[0]):
        for i in range(motion_quaternions.shape[1]):
            if dot_quaternion(motion_quaternions[j,i],motion_quaternions[j,0]) < 0:
                motion_quaternions[j,i] = -motion_quaternions[j,i]               
            if dot_quaternion(motion_quaternions[j,i],motion_quaternions[j,0]) < 0:
                print (j,i)
                pdb.set_trace() 


    t_in = np.array(t_in)

    t_out = np.array(range(motion_thetas.shape[0]))
    
    #print ("start smooth")

    #pdb.set_trace()

    motion_quaternions_result = [] 

    for j in range(joints_number): 
        motion_quaternions_result.append(quaternion.quaternion_time_series.squad(motion_quaternions[j],t_in,t_out))   
    #print ("end smooth")

    motion_quaternions = np.array(motion_quaternions_result)

    theta_final = [[] for i in range(motion_quaternions.shape[1])]

    for i in range(motion_quaternions.shape[1]):
        for j in range(motion_quaternions.shape[0]):
            theta_final[i].append(_cv2.Rodrigues(quaternion.as_rotation_matrix(motion_quaternions[j,i]))[0].ravel())
 
    theta_final = np.reshape(np.array(theta_final),(-1,1,3*joints_number))

    #pdb.set_trace()

    return theta_final
        
def buid_graph_body_2D(joints_restri,joints_restri_x,joints_restri_y,joint_achor,trans_0,pose_0,joints_0,renderer,pose_mean, pose_covariance,model_path):
    
    tf.reset_default_graph() 
    g_2 = tf.Graph()
    
    config = flags.FLAGS

    with g_2.as_default(): 
       betas = tf.placeholder(dtype = tf.float32, shape = (trans_0.shape[0],10))  
       prior_pose = tf.placeholder(dtype = tf.float32, shape = ())

       #pose = tf.Variable(pose_0,dtype = tf.float32)
       #pose_new = tf.concat([pose, tf.zeros((1,69))], axis=1) 
       pose_new = tf.Variable(pose_0,dtype = tf.float32)
       trans_new = tf.constant(trans_0,dtype = tf.float32)
       pose = tf.constant(pose_0,dtype = tf.float32)
       #trans = tf.constant(trans_0,dtype = tf.float32)
       joints = tf.constant(joints_0,dtype = tf.float32)
       
       pose_mean = tf.constant(pose_mean, dtype=tf.float32)
       pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)       
       model = SMPL(model_path, joint_type=config.joint_type)
       Js = model(betas,pose_new,get_skin=False)
       
       joints_new = model.J_transformed + tf.stack([tf.stack([trans_new[i,:] for j in range(24)]) for i in range(trans_0.shape[0])])
       
       keypoints_new = tf.stack([renderer.openpose_joints(tf.reshape(model.J_transformed[i,:,:],(1,-1,3)),trans_new[i,:]) for i in range(trans_0.shape[0])])

   
       pose_diff = tf.stack( [tf.reshape(pose_new[i,:,3:] - pose_mean, [1, -1]) for i in range(trans_0.shape[0])])

      
       loss_restriction_3d = tf.zeros((1,1),dtype = tf.float32)
       loss_restriction_2d = tf.zeros((1,1),dtype = tf.float32)
       loss_prior = tf.zeros((1,1),dtype = tf.float32)
       loss_sim = tf.zeros((1,1),dtype = tf.float32)
       
       for i,jrs in enumerate(joints_restri):

           for t in joint_achor[i]:
               loss_restriction_3d = tf.concat([loss_restriction_3d,tf.reshape(tf.square(joints[i,t,:] - joints_new[i,t,:] ),[-1,3])],1)
               loss_sim = tf.concat([loss_sim,tf.reshape(tf.square(pose[i,0,3*t:3*t+3] - pose_new[i,0,3*t:3*t+3] ),[-1,3])],1)
                                  
           for ij,j in enumerate(jrs):
               loss_restriction_2d = tf.concat([loss_restriction_2d,tf.reshape(tf.square(keypoints_new[i,j,:] - tf.constant([joints_restri_x[i][ij],joints_restri_y[i][ij]],dtype = tf.float32)),[-1,2])],1)
          
           loss_prior = tf.concat([loss_prior,tf.matmul(tf.matmul(pose_diff[i,:], pose_covariance), tf.transpose(pose_diff[i,:]))],1)


       loss_2d = tf.reduce_sum(loss_restriction_2d)# + 1e-4*tf.reduce_sum(loss_prior)
       loss_3d = 1e+5*(tf.reduce_sum(loss_restriction_3d) + tf.reduce_sum(loss_sim)  )
       loss_p = prior_pose*tf.reduce_sum(loss_prior)

       loss_restriction = loss_3d + 1e+1*loss_2d

       loss = loss_restriction + loss_p

       error = tf.train.AdamOptimizer(1e-2).minimize(loss)

       return g_2,betas,pose_new,trans_new,joints_new,error,loss_3d,loss_2d,loss_p,loss,prior_pose,loss




def retargeting_otimization_2D(joints,trans,my_shape,flength,motion_betas,motion_model_type,motion_thetas,all_restrictions):

    
    config = flags.FLAGS
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path,frame_width=my_shape[1],frame_height=my_shape[0],flength=flength)    
    pose_mean, pose_covariance = MuVs_util.load_initial_param()

        
    joint_number = {20:0,21:1,7:2,8:3}

    free_joints = [[13,16,18,20,22],[14,17,19,21,23],[1,4,7,10],[2,5,8,11]]

    tree_joints = [[3,6,9,13,16,18,20,22],[3,6,9,14,17,19,21,23],[1,4,7,10],[2,5,8,11]]
         
    frames = []
    anchors = []
    joints_restri = []

    joints_restri_x = [] 
    joints_restri_y = [] 

    trans_0 = [] 
    pose_0 = []
    joints_0 = []

    joint_achor = [] 

    for restri in all_restrictions:
        if restri.my_type == 1:
            for i,f in enumerate(restri.restriction_point):
                if f in frames:
                    index = frames.index(f)
                    joints_restri[index].append(restri.joint)
                    joints_restri_x[index].append(restri.x[i])
                    joints_restri_y[index].append(restri.y[i])

                else:
                    frames.append(f)
                    index = len(frames) -1
                    joints_restri.append([restri.joint])
                    joints_restri_x.append([restri.x[i]])
                    joints_restri_y.append([restri.y[i]])
                
    if len(frames) == 0:
        return joints,trans,motion_thetas                                 
  
    

    for i,f in enumerate(frames):
        joint_achor.append(list(range(24)))
        for j in joints_restri[i]:
            for k in free_joints[joint_number[j]]:                
                joint_achor[i].remove(k)

        trans_0.append(trans[f])
        pose_0.append(motion_thetas[f])
        joints_0.append(joints[f])

    #print (motion_thetas.shape)
   
   

    trans_0 = np.array(trans_0)
    pose_0 = np.array(pose_0)
    joints_0 = np.array(joints_0) 

    #print (pose_0.shape)

    if motion_model_type == 1:
        model_path = config.smpl_model_path_m
    elif motion_model_type == 2: 
        model_path = config.smpl_model_path_f 
    else: 
        model_path = config.smpl_model_path                


    g_2,betas_in,pose_new,trans_new,joints_new,error,loss_3d,loss_2d,loss_prior,loss,prior_pose,loss =  buid_graph_body_2D(joints_restri,joints_restri_x,joints_restri_y,joint_achor,trans_0,pose_0,joints_0,renderer,pose_mean, pose_covariance,model_path)


    with tf.Session(config=config_tf,graph=g_2) as sess:
        sess.run(tf.global_variables_initializer())
        for s in range(300):
            pose_resposta = sess.run([pose_new,trans_new,joints_new,error,loss_3d,loss_2d,loss_prior,loss], feed_dict = {betas_in:np.stack([motion_betas for i in range(len(frames))]),prior_pose:1.0e-2})
            #print (str(pose_resposta[4]) + " " + str(pose_resposta[5]) + " " + str(pose_resposta[6]) + " " + str(pose_resposta[7]))
        
    sess.close()


    for i,f in enumerate(frames):
        #pdb.set_trace()
        joints[f] = pose_resposta[2][i]
        trans[f] =  pose_resposta[1][i]  
        motion_thetas[f] = pose_resposta[0][i]
    
    for i in range(24):
        tmp_thetas = motion_thetas[:,:,3*i:3*i+3]
        to_use = to_use = [True for x in range(motion_thetas.shape[0])]

        for restri in all_restrictions:
            if restri.my_type == 1:
                for p in range(restri.start,restri.end):
                    if (i in tree_joints[joint_number[restri.joint]]) and not (p in restri.restriction_point):
                        to_use[p] = False
        
        motion_thetas[:,:,3*i:3*i+3] = quaternion_interpolation(tmp_thetas,to_use,joints_number=1)
    
    print (motion_thetas.shape)


    g_1,Betas,Thetas,Trans,Joints =  smpl_regression(model_path)

    joints = [[] for _ in range(motion_thetas.shape[0])]
       
    with tf.Session(config=config_tf,graph=g_1) as sess:
           sess.run(tf.global_variables_initializer())
           for i in range(motion_thetas.shape[0]):
               pose_resposta_2 = sess.run([Joints], feed_dict = {Thetas:motion_thetas[i],Betas:np.reshape(motion_betas,(-1,10)),Trans:np.reshape(trans[i],(-1,3))})
               joints[i].append(pose_resposta_2[0])
                 

    sess.close()

  
    joints = np.reshape(np.array(joints),(-1,24,3))

   
    return joints,trans,motion_thetas

def buid_graph_body_3D(trans_0,pose_0,joints_0,pose_mean, pose_covariance,model_path,all_restrictions,start,end):
    
    tf.reset_default_graph() 
    g_2 = tf.Graph()
    
    config = flags.FLAGS

    with g_2.as_default(): 
       betas = tf.placeholder(dtype = tf.float32, shape = (trans_0.shape[0],10))  
       prior_pose = tf.placeholder(dtype = tf.float32, shape = ())

       #pose = tf.Variable(pose_0,dtype = tf.float32)
       #pose_new = tf.concat([pose, tf.zeros((1,69))], axis=1) 
       pose_new = tf.Variable(pose_0,dtype = tf.float32)
       trans_new = tf.Variable(trans_0,dtype = tf.float32)
       pose = tf.constant(pose_0,dtype = tf.float32)
       trans = tf.constant(trans_0,dtype = tf.float32)
       joints = tf.constant(joints_0,dtype = tf.float32)

       sim_weigths_0 = [1.0,2.0,2.0,1.0,3.0,3.0,1.0,4.0,4.0,1.0,5.0,5.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,4.0,5.0,5.0,6.0,6.0]*20

       sim_weigths = tf.constant([ 1.0/(sim_weigths_0[int(x/3)]*sim_weigths_0[int(x/3)]) for x in range(72)],dtype = tf.float32)
       
       pose_mean = tf.constant(pose_mean, dtype=tf.float32)
       pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)       
       model = SMPL(model_path, joint_type=config.joint_type)
       Js = model(betas,pose_new,get_skin=False)
       
       joints_new = model.J_transformed + tf.stack([tf.stack([trans_new[i,:] for j in range(24)]) for i in range(trans_0.shape[0])])
       
   
       pose_diff = tf.stack( [tf.reshape(pose_new[i,:,3:] - pose_mean, [1, -1]) for i in range(trans_0.shape[0])])

      
       loss_restriction_3d = tf.zeros((1,1),dtype = tf.float32)
       loss_prior = tf.zeros((1,1),dtype = tf.float32)
       loss_sim = tf.zeros((1,1),dtype = tf.float32)
       
       for i in range(trans_0.shape[0]):

           for restri in all_restrictions:
               if restri.my_type == 0:
                   if len(restri.restriction_point) == 0: 
                       restri.restriction_point = list(range(restri.start,restri.end+1))

                   if (i + start) in restri.restriction_point:
                       loss_restriction_3d = tf.concat([loss_restriction_3d,tf.reshape(tf.square(joints[i,restri.joint,:] - joints_new[i,restri.joint,:] ),[-1,3])],1)
               
           
           loss_sim = tf.concat([loss_sim,tf.reshape(sim_weigths*tf.square(pose[i,:,:] - pose_new[i,:,:]),[1,-1])],1)      
          
           loss_prior = tf.concat([loss_prior,tf.matmul(tf.matmul(pose_diff[i,:], pose_covariance), tf.transpose(pose_diff[i,:]))],1)

       loss_trans = tf.reduce_sum(tf.square(trans[:,0] - trans_new[:,0])) + tf.reduce_sum(tf.square(trans[:,2] - trans_new[:,2]))

       loss_3d = 1e+2*tf.reduce_sum(loss_restriction_3d) + tf.reduce_sum(loss_sim) + 0.5*1e+2*loss_trans
       loss_p = prior_pose*tf.reduce_sum(loss_prior)

       loss = loss_3d + loss_p

       error = tf.train.AdamOptimizer(1e-2).minimize(loss)

       return g_2,betas,pose_new,trans_new,joints_new,error,loss_3d,loss_p,loss,prior_pose,loss




def retargeting_otimization_3D(joints,trans,source_betas,source_model_type,motion_thetas,all_restrictions,smoothing=0.8):

    end_effector = [20,21,7,8]

    config = flags.FLAGS
    pose_mean, pose_covariance = MuVs_util.load_initial_param()
    
    if source_model_type == 1:
        model_path = config.smpl_model_path_m
    elif source_model_type == 2: 
        model_path = config.smpl_model_path_f 
    else: 
        model_path = config.smpl_model_path 

    print ("IK to inicialization") 

    window = 40

    for i in range(0,joints.shape[0],window):  
        print ("Doing ",i)
            
        g_2,betas_in,pose_new,trans_new,joints_new,error,loss_3d,loss_sim,loss_prior,prior_pose,loss =  buid_graph_body_3D(trans[i:np.minimum(i+window,joints.shape[0])],motion_thetas[i:np.minimum(i+window,joints.shape[0])],joints[i:np.minimum(i+window,joints.shape[0])],pose_mean, pose_covariance,model_path,all_restrictions,i,np.minimum(i+window,joints.shape[0] -1))

        

        with tf.Session(config=config_tf,graph=g_2) as sess:
            sess.run(tf.global_variables_initializer())
            for s in range(300):
                pose_resposta = sess.run([pose_new,trans_new,joints_new,error,loss_3d,loss_sim,loss_prior,loss], feed_dict = {betas_in:np.reshape(np.array([source_betas for i in range((joints[i:np.minimum(i+window,joints.shape[0])]).shape[0])]),(-1,10)),prior_pose:1.0e-4})
            #print (str(pose_resposta[4]) + " " + str(pose_resposta[5]) + " " + str(pose_resposta[6]) + " " + str(pose_resposta[7]))
        
            trans[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[1]
            motion_thetas[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[0]
            joints[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[2]

        sess.close()


    time = np.arange(joints.shape[0])


    

    for joint in end_effector:
        w = np.ones_like(time)*1.0
        for t in time:
            for restri in all_restrictions:
                if restri.my_type == 0:
                    if restri.joint == joint:
                        if t > restri.start and t < restri.end:
                            w[t] = 0.5
                        if t == restri.start or t == restri.end:
                            w[t] = 50.0
                        if len(restri.restriction_point) == 0 or (t in restri.restriction_point):
                            w[t] = 100.0
                     
        #pdb.set_trace()
        joints_sm = np.transpose(csaps(time, np.transpose(joints[:,joint,:]), time,weights=w,smooth=smoothing)) 

        for i,wi in enumerate(w):
            if wi < 1.0:
                joints[i,joint,:] = joints_sm[i,:]



    
    print ("IK adjustment") 


    all_restrictions_now = [Restriction(my_type=0,start=0,end=joints.shape[0] +1 ,joint=7),
                            Restriction(my_type=0,start=0,end=joints.shape[0] +1,joint=8),
                            Restriction(my_type=0,start=0,end=joints.shape[0] +1,joint=20),
                            Restriction(my_type=0,start=0,end=joints.shape[0] +1,joint=21)]

    
    

    for i in range(0,joints.shape[0],window):  
        print ("Doing ",i)
            
        g_2,betas_in,pose_new,trans_new,joints_new,error,loss_3d,loss_sim,loss_prior,prior_pose,loss =  buid_graph_body_3D(trans[i:np.minimum(i+window,joints.shape[0])],motion_thetas[i:np.minimum(i+window,joints.shape[0])],joints[i:np.minimum(i+window,joints.shape[0])],pose_mean, pose_covariance,model_path,all_restrictions_now,i,np.minimum(i+window,joints.shape[0] -1))        

        with tf.Session(config=config_tf,graph=g_2) as sess:
            sess.run(tf.global_variables_initializer())
            for s in range(300):
                pose_resposta = sess.run([pose_new,trans_new,joints_new,error,loss_3d,loss_sim,loss_prior,loss], feed_dict = {betas_in:np.reshape(np.array([source_betas for i in range((joints[i:np.minimum(i+window,joints.shape[0])]).shape[0])]),(-1,10)),prior_pose:1.0e-4})
            #print (str(pose_resposta[4]) + " " + str(pose_resposta[5]) + " " + str(pose_resposta[6]) + " " + str(pose_resposta[7]))
        
            trans[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[1]
            motion_thetas[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[0]
            joints[i:np.minimum(i+window,joints.shape[0])] = pose_resposta[2]

        sess.close()



    return trans,motion_thetas



def retargeting_otimization(joints,trans,my_shape,flength,motion_betas,motion_model_type,motion_thetas,all_restrictions,source_model_type,source_betas,smoothing=0.01):

  
    print ("Otimization to 2D restrictions")
    joints,trans,motion_thetas = retargeting_otimization_2D(joints,trans,my_shape,flength,motion_betas,motion_model_type,motion_thetas,all_restrictions)

    
    print ("Otimization to 3D restrictions ")

    trans, motion_thetas = retargeting_otimization_3D(joints,trans,source_betas,source_model_type,motion_thetas,all_restrictions,smoothing=smoothing)

     
    return trans, motion_thetas






