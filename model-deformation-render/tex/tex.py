import sys as _sys
import os as _os
import os.path as _path
import numpy as np
import cv2
import cPickle as pickle
from copy import copy
from pyoctree import pyoctree as _pyoctree
import chumpy as ch

import pdb

_sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
from config import SMPL_FP
_sys.path.insert(0, SMPL_FP)
try:
    # Robustify against setup.
    from smpl.serialization import load_model as _load_model
except ImportError:
    # pylint: disable=import-error
    try:
        from psbody.smpl.serialization import load_model as _load_model
        from psbody.smpl.lbs import global_rigid_transformation as _global_rigid_transformation
        from psbody.smpl.verts import verts_decorated
        from psbody.smpl.lbs import verts_core as _verts_core
         
    except:
        from smpl_webuser.serialization import load_model as _load_model
        from smpl_webuser.lbs import global_rigid_transformation as _global_rigid_transformation
        from smpl_webuser.lbs import verts_core as _verts_core

from up_tools.mesh import Mesh

from src.draw import warp_tri

_TEMPLATE_MESH = Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                             '..', 'models', '3D', 'template.ply'))
_TEX_MAP_FNAME = _os.path.join(
    _os.path.dirname(__file__), '..', 'models', 'mapeamento2.obj')

_MOD_PATH = _os.path.abspath(_os.path.dirname(__file__))
# Models:
# Models:


_MODEL_NEUTRAL_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D',
    'lbs_tj10smooth6_0fixed_normalized_locked_hybrid_model_novt.pkl')

_MODEL_MALE_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D',
    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

_MODEL_FEME_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D',
    'basicModel_f_lbs_10_207_0_v1.0.0.pkl')

_MODEL_N = _load_model(_MODEL_NEUTRAL_FNAME)
_MODEL_M = _load_model(_MODEL_MALE_FNAME)
_MODEL_F = _load_model(_MODEL_FEME_FNAME)

_MODEL_VT = _os.path.join(
    _MOD_PATH, '..', 'models','basicModel_vt.npy')

_MODEL_FT = _os.path.join(
    _MOD_PATH, '..', 'models','basicModel_ft.npy')


_MODEL_VT_FT_OBJ = _os.path.join(
    _MOD_PATH, '..', 'models','mapeamento_impaint_2.obj')

def Template_tex():

         #mesh.vt,mesh.ft = Template_tex()  
        vt = np.load(_MODEL_VT)
        ft = np.load(_MODEL_FT)

	return vt,ft


def Template_tex_from_obj(file_path):

        vt = []
        ft = []
       
        with open(file_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:

                if line[:2] == "vt" :
                    values = line.split(' ')
                    vt.append([float(values[1]),float(values[2])])

                elif line[0] == 'f' : 
                    values = line.split(' ')
                    ft_now = []
                    for v_in_f in values[1:]:
                        v_vt_n = v_in_f.split("/")
                        if len(v_vt_n) < 2:
                            print ("obj file does not have texture information")
                            exit()                        

                        ft_now.append(int(v_vt_n[1]) -1 ) # obj is 1 index start
                    
                    ft.append(ft_now)

                line = fp.readline()
                cnt += 1
                
                    
        vt = np.array(vt)
        ft = np.array(ft)
               

	return vt,ft


def dist(d1,d2):
	d01 = np.array(d1)
	d02 = np.array(d2)
	return np.sqrt(np.sum(np.power(d01 - d02,2)))

def texture_vis(model, image, data,v_t,mask):

    mesh = copy(_TEMPLATE_MESH)
 
    my_tex = np.zeros((1000,1000,image.shape[2]),dtype=image.dtype)

    #cmesh = Mesh(_os.path.join(_os.path.dirname(__file__),
    #                           'template-bodyparts-corrected-labeled-split5.ply'))
    #mesh.vc = cmesh.vc.copy()

    mesh.vc = [0.91, 0.67, 0.91]   
    model.betas[:len(data['betas'])] = data['betas']
    model.pose[:] = data['pose']      
    model.trans[:] = data['trans'] 

    mesh.v = v_t

    #pdb.set_trace()

    mesh.v =  np.array(_verts_core(model.pose, mesh.v, model.J, model.weights, model.kintree_table)) 

    #pdb.set_trace()

    #new_v_t =  _verts_core(model.pose, v_t, model.J, model.weights, model.kintree_table)   
    

    #mesh.v = mesh.v + np.array(new_v_t) 

    largura, altura = (image.shape[1], image.shape[0])

    data['cam_c'] = np.array([largura, altura]) / 2.

    tex_vt,tex_faces = Template_tex()

    vert = mesh.v + np.array(data['trans']).reshape((1,3))

    faces = np.array(mesh.f.copy(),dtype='int32')
    tree = _pyoctree.PyOctree(vert,faces)

    cp = np.zeros((vert.shape[0],4))

    #my_map_partes = np.zeros(vert.shape[0]).astype(int)
 
    for i,f in enumerate(mesh.f):

        #if ((i < 300) or (i > 5046)):
        #    continue

        I_VIS = True

        face_media = (vert[f[0]] + vert[f[1]] + vert[f[2]])/3.0

        rayList = np.array([np.zeros(3),face_media],dtype=np.float32)
	intersectionFound = tree.rayIntersection(rayList)

        if intersectionFound:
            if dist(intersectionFound[0].p,face_media) > 0.0001:
                I_VIS = False
         
        if I_VIS:

             tri1 = np.zeros((3,2),np.float32)
             tri2 = np.zeros((3,2),np.float32)

             tri1[0,0] = (vert[f[0]][1]*data['f'])/(vert[f[0]][2]) + data['cam_c'][1]

             tri1[0,1] = (vert[f[0]][0]*data['f'])/(vert[f[0]][2]) + data['cam_c'][0]
             
             tri1[1,0] = (vert[f[1]][1]*data['f'])/(vert[f[1]][2]) + data['cam_c'][1]

             tri1[1,1] = (vert[f[1]][0]*data['f'])/(vert[f[1]][2]) + data['cam_c'][0]

             tri1[2,0] = (vert[f[2]][1]*data['f'])/(vert[f[2]][2]) + data['cam_c'][1]

             tri1[2,1] = (vert[f[2]][0]*data['f'])/(vert[f[2]][2]) + data['cam_c'][0]
          

             cp[f[0]][0] = 1.0
             cp[f[0]][1] = tri1[0,0]
             cp[f[0]][2] = tri1[0,1]
             cp[f[0]][3] = vert[f[0]][2]

             cp[f[1]][0] = 1.0
             cp[f[1]][1] = tri1[1,0]
             cp[f[1]][2] = tri1[1,1]
             cp[f[1]][3] = vert[f[1]][2]

             cp[f[2]][0] = 1.0
             cp[f[2]][1] = tri1[2,0]
             cp[f[2]][2] = tri1[2,1]
             cp[f[2]][3] = vert[f[2]][2]


             tri2[0,1] = (tex_vt[tex_faces[i][0]][0])*1000
             tri2[0,0] = 1000 -  (tex_vt[tex_faces[i][0]][1])*1000

             tri2[1,1] = (tex_vt[tex_faces[i][1]][0])*1000
             tri2[1,0] = 1000 - (tex_vt[tex_faces[i][1]][1])*1000

             tri2[2,1] = (tex_vt[tex_faces[i][2]][0])*1000
             tri2[2,0] = 1000 - (tex_vt[tex_faces[i][2]][1])*1000

             '''my_map_partes[f[0]] = (image[int(tri2[0,0]),int(tri2[0,1]),0])/15 if  (image[int(tri2[0,0]),int(tri2[0,1]),0])/15 > 0 else my_map_partes[f[0]] 
             my_map_partes[f[1]] = (image[int(tri2[1,0]),int(tri2[1,1]),0])/15 if  (image[int(tri2[1,0]),int(tri2[1,1]),0])/15 > 0 else my_map_partes[f[1]] 
             my_map_partes[f[2]] = (image[int(tri2[2,0]),int(tri2[2,1]),0])/15 if  (image[int(tri2[2,0]),int(tri2[2,1]),0])/15 > 0 else my_map_partes[f[2]] '''

             #if image[int(tri2[0,0]),int(tri2[0,1]),0] == 15:
             #warp_tri(tri2,tri2,image,my_tex)

             #tex_vt[tex_faces[i][0] - 1][1]

             #print tri1

             

             warp_tri(tri1,tri2,image,my_tex)

             
             

             #cv2.imshow('ImageWindow', my_tex)
             #cv2.waitKey()	
 

    #np.savetxt('test.txt',  my_map_partes , delimiter='\n') 

    return my_tex,cp

def texture_remap(image, direction=True):

    #print (image.shape)
    
    my_tex = np.zeros((1000,1000,image.shape[2]),dtype=image.dtype)
    
    largura, altura = (image.shape[1], image.shape[0])

    if direction:
        tex_vt_in,tex_faces_in = Template_tex()
        tex_vt_out,tex_faces_out = Template_tex_from_obj(_MODEL_VT_FT_OBJ)
        #print ("Ok")
    else:
        tex_vt_in,tex_faces_in = Template_tex_from_obj(_MODEL_VT_FT_OBJ)
        tex_vt_out,tex_faces_out = Template_tex()
 
    #pdb.set_trace()

    
    for i in range(tex_faces_in.shape[0]):
        tri1 = np.zeros((3,2),np.float32)
        tri2 = np.zeros((3,2),np.float32) 

        tri1[0,1] = (tex_vt_in[tex_faces_in[i][0]][0])*1000
        tri1[0,0] = 1000 -  (tex_vt_in[tex_faces_in[i][0]][1])*1000

        tri1[1,1] = (tex_vt_in[tex_faces_in[i][1]][0])*1000
        tri1[1,0] = 1000 - (tex_vt_in[tex_faces_in[i][1]][1])*1000

        tri1[2,1] = (tex_vt_in[tex_faces_in[i][2]][0])*1000
        tri1[2,0] = 1000 - (tex_vt_in[tex_faces_in[i][2]][1])*1000  

       
        tri2[0,1] = (tex_vt_out[tex_faces_out[i][0]][0])*1000
        tri2[0,0] = 1000 -  (tex_vt_out[tex_faces_out[i][0]][1])*1000

        tri2[1,1] = (tex_vt_out[tex_faces_out[i][1]][0])*1000
        tri2[1,0] = 1000 - (tex_vt_out[tex_faces_out[i][1]][1])*1000

        tri2[2,1] = (tex_vt_out[tex_faces_out[i][2]][0])*1000
        tri2[2,0] = 1000 - (tex_vt_out[tex_faces_out[i][2]][1])*1000  
        
        warp_tri(tri1,tri2,image,my_tex)    

    return my_tex


def test_1():
    model = _MODEL_F
    with open("GOPR95340000000000_IUV.png_body.pkl") as f:
        data = pickle.load(f)
    image = cv2.imread("GOPR95340000000000.png")
    mask = cv2.imread("GOPR95340000000000_seg.png",cv2.IMREAD_GRAYSCALE)
    _, silh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    v_t = np.zeros((6890,3))

    model.betas[:len(data['betas'])] = data['betas']
    model.pose[:] = data['pose']      
    model.trans[:] = data['trans'] 

    tex,cp = texture_vis(model, image, data,model.v_posed,silh.astype(np.uint8))
    
    cv2.imshow('ImageWindow', tex)
    cv2.waitKey()
    

def test_2():

    image = cv2.imread("tex_saida_0.png")
  
    tex = texture_remap(image)
    
    cv2.imshow('ImageWindow', tex)
    cv2.waitKey()
    
 

if __name__ == '__main__':
    test_2()
   






