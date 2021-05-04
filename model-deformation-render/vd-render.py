#!/usr/bin/env python

"""Model deformation and render, simple use: python vd-render.py ../data/box/ ../data/8-views/ ../data/out/ --model_type 2"""

# pylint: disable=invalid-name
import sys as _sys
import os as _os
import subprocess
import os.path as _path
import logging as _logging
import cPickle as pickle
from copy import copy
import glob as _glob

import numpy as np
import chumpy as ch
import cv2
import click as _click
import matplotlib.pyplot as plt

from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
#_sys.path.append('./')

from natsort import natsorted

_FLENGTH_GUESS = 1500. #2500.

from up_tools.mesh import Mesh
from up_tools.mesh import write_obj
from up_tools.camera import rotateY,translateG
_sys.path.insert(0, _path.join(_path.dirname(__file__), '.'))
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

_LOGGER = _logging.getLogger(__name__) 
_TEMPLATE_MESH = Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                             '.', 'models', '3D', 'template.ply'))
_TEX_MAP_FNAME = _os.path.join(
    _os.path.dirname(__file__), '.', 'models', 'mapeamento2.obj')

_MOD_PATH = _os.path.abspath(_os.path.dirname(__file__))
# Models:
# Models:

_MODEL_MAP_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models','map_partes_14.txt')

_TEX_P_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models','map_tex_partes_new.png')


_TEX_A_POSE_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models','A-pose_map.png')

_TEX_P_FNAME_CNN = _os.path.join(
    _MOD_PATH, '.', 'models','map_tex_partes_cnn_new.png')

_MODEL_NEUTRAL_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'lbs_tj10smooth6_0fixed_normalized_locked_hybrid_model_novt.pkl')

_MODEL_MALE_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

_MODEL_FEME_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'basicModel_f_lbs_10_207_0_v1.0.0.pkl')

_MODEL_VT = _os.path.join(
    _MOD_PATH, '.', 'models','basicModel_vt.npy')

_MODEL_FT = _os.path.join(
    _MOD_PATH, '.', 'models','basicModel_ft.npy')


from src.visibility_match import visibility_match
from src.draw import warp_tri
from deformation import deformation
from normal import compute_normals
from tex.tex import texture_vis
from tex.tex import texture_remap
from src.feather_blending import feather_blending

import pdb

pose_partes = np.genfromtxt(_MODEL_MAP_FNAME)

size_partes = [np.count_nonzero(pose_partes == x) for x in range(15)] 


tex_map_partes = cv2.imread(_TEX_P_FNAME,0)


tex_A_pose = cv2.imread(_TEX_A_POSE_FNAME,0)

tex_map_partes_cnn = cv2.imread(_TEX_P_FNAME_CNN,0)

#print tex_map_partes.shape

#pdb.set_trace()
#print pose_partes

_MODEL_N = _load_model(_MODEL_NEUTRAL_FNAME)
_MODEL_M = _load_model(_MODEL_MALE_FNAME)
_MODEL_F = _load_model(_MODEL_FEME_FNAME)

_COLORS = {
    'pink': [0.91, 0.67, 0.91],
    'cyan': [.7, .75, .5],
    'yellow': [.5, .7, .75],
    'black': [.0, .0, .0],
}


label_colours = [(0,0,0,255), (128,0,0,255), (255,0,0,255), (0,85,0,255), (170,0,51,255), (255,85,0,255), (0,0,85,255), (0,119,221,255), (85,85,0,255), (0,85,85,255), (85,51,0,255), (52,86,128,255), (0,128,0,255),(0,0,255,255), (51,170,221,255), (0,255,255,255), (85,255,170,255), (170,255,85,255), (255,255,0,255), (255,170,0,255)]


label_p_names_seg = {"back":0,"head":1, "lfoot":2, "lhand":3,"llarm":4,"llleg":5,"luarm":6,"luleg":7,
                    "rfoot":8,"rhand":9,"rlarm":10,"rlleg":11,"ruarm":12,"ruleg":13,"torso":14}


label_p_names_model = {"back":0,"head":1,"torso":2,"luarm":3,"ruarm":4,"llarm":5,"rlarm":6,"luleg":7,
                       "ruleg":8,"llleg":9,"rlleg":10,"lfoot":11,"rfoot":12,"lhand":13,"rhand":14}

label_map_gt = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210]

label_map_cnn = [0,1,14,6,12,4,10,7,13,5,11,2,8,3,9]



limit_deformation = [0.08]*15

limit_deformation[2] = 0.12
#limit_deformation[11] = 0.04
#limit_deformation[12] = 0.04
limit_deformation[13] = 0.02
limit_deformation[14] = 0.02


print limit_deformation

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def distancia(a,b):
    return  np.sqrt((a[0] -b[0])*(a[0] -b[0]) + (a[1] -b[1])*(a[1] -b[1]) + (a[2] -b[2])*(a[2] -b[2]))
 


def magnitude(vector):
   return np.sqrt(np.dot(np.array(vector),np.array(vector)))

def norm(vector):
   return np.array(vector)/magnitude(np.array(vector))


def lineRayIntersectionPoint(rayOrigin, rayDirection, point1, point2):
    # Convert to numpy arrays
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array(rayDirection, dtype=np.float) - rayOrigin    
    my_mag = magnitude(rayDirection)

    rayDirection = np.array(norm(rayDirection), dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)

    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 1.0*my_mag and t2 >= 0.0 and t2 <= 1.0:
        #pdb.set_trace()
        return True
    return False


def intersect_is_valid(s,p,my_segments,point0,point1):
    for s0,c in enumerate(my_segments):
        for p0,point in enumerate(c):
            n_point = (p0 +1)%len(c)
            if not (s0 == s and (p0 == p or n_point == p)):
                #pdb.set_trace()
                if lineRayIntersectionPoint(point0[:-1], point1, point[0], c[n_point][0]):
                    #print point0[:-1], point1, point[0], c[n_point][0]
                    #pdb.set_trace()
                    return False 
                               
                    

    return True


def Template_tex():

         #mesh.vt,mesh.ft = Template_tex()  
        vt = np.load(_MODEL_VT)
        ft = np.load(_MODEL_FT)

	return vt,ft


def _create_renderer(  # pylint: disable=too-many-arguments
        w=640,
        h=480,
        rt=np.zeros(3),
        t=np.zeros(3),
        f=None,
        c=None,
        k=None,
        near=1.,
        far=10.,
        texture=None):
    """Create a renderer for the specified parameters."""
    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5)           if k is None else k

    if texture is not None:
        rn = TexturedRenderer()
    else:
        rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}
    if texture is not None:
        rn.texture_image = np.asarray(cv2.imread(texture), np.float64)/255.
        rn.texture_image = rn.texture_image[:,:,::-1]
        #print rn.texture_image.shape
    return rn

def _create_renderer_mesh(  # pylint: disable=too-many-arguments
        w=640,
        h=480,
        rt=np.zeros(3),
        t=np.zeros(3),
        f=None,
        c=None,
        k=None,
        near=1.,
        far=10.,
        texture=None):
    """Create a renderer for the specified parameters."""
    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5)           if k is None else k

    if texture is not None:
        rn = TexturedRenderer()
    else:
        rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}

    if texture is not None:
        rn.texture_image = np.asarray(texture, np.float64)/255.
        rn.texture_image = rn.texture_image[:,:,::-1]
        #print rn.texture_image.shape
    return rn



def _stack_with(rn, mesh, texture):
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            mesh.ft = mesh.f
            mesh.vt = mesh.v[:, :2]
        rn.ft = np.vstack((rn.ft, mesh.ft+len(rn.vt)))
        rn.vt = np.vstack((rn.vt, mesh.vt))
    rn.f = np.vstack((rn.f, mesh.f+len(rn.v)))
    rn.v = np.vstack((rn.v, mesh.v))
    rn.vc = np.vstack((rn.vc, mesh.vc))


def _simple_renderer(rn, meshes, yrot=0, texture=None,out_mesh_name=None,out_texture_name=None):
    mesh = meshes[0]
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            """mesh.ft = copy(mesh.f)
            vt = copy(mesh.v[:, :2])
            vt -= np.min(vt, axis=0).reshape((1, -1))
            vt /= np.max(vt, axis=0).reshape((1, -1))
            mesh.vt = vt"""
            mesh.vt,mesh.ft = Template_tex()

        mesh.texture_filepath = rn.texture_image
  
    # Set camers parameters
    if texture is not None:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, ft=mesh.ft, vt=mesh.vt, bgcolor=np.ones(3))
    else:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, bgcolor=np.ones(3))

    for next_mesh in meshes[1:]:
        _stack_with(rn, next_mesh, texture)    
    
    if out_mesh_name is not None:
        write_obj(mesh,out_mesh_name)

    if (texture is not None) and (out_texture_name is not None):
        cv2.imwrite(out_texture_name,texture)
        with open(out_mesh_name[:-3] + "mtl", 'w') as my_file:
            my_file.write("newmtl Material.001" + "\n" + "Ns 96.078431" + "\n" + "Ka 1.000000 1.000000 1.000000" + "\n" + "Kd 0.640000 0.640000 0.640000" + "\n" + "Ks 0.000000 0.000000 0.000000" + "\n" + "Ke 0.412673 0.432000 0.226290" + "\n" + "Ni 1.000000" + "\n" + "d 1.000000" + "\n" + "illum 1" + "\n" + "map_Kd " +  out_texture_name)
   
    # Construct Back Light (on back right corner)
 
    if texture is not None:
        rn.vc = ch.ones(rn.v.shape)
    else:
        albedo = rn.vc
    # Construct Back Light (on back right corner)
        rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
        rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
        rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))
    
    return rn.r


# pylint: disable=too-many-locals
def get_mesh_to_deformation(filename,model,cam,label_folder):

    image = cv2.imread(filename,-1)    
    #if image.shape[2] != 4:
    #    print "No segmented image "
    #    exit()      

    mesh = copy(_TEMPLATE_MESH)
    mesh.vc = _COLORS['pink']
    # render ply
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']    
    model.trans[:] = cam['trans']
    mesh.v = model.r

    # transformation for each v [4,4,6890]

    inv_trasformation = np.zeros((4,4,6890))

    (A, A_global) = _global_rigid_transformation(model.pose, model.J, model.kintree_table, xp=ch)
    T = A.dot(model.weights.T)
    T0 = np.array(T)

    for i in range(6890):
         m_inv = np.linalg.inv(T0[:,:,i])
         inv_trasformation[:,:,i] = m_inv
         #inv_trasformation.append(m_inv.tolist())

    #pdb.set_trace()

    contor_list = []
    list_path = filename.split("/")
    path = ''.join(e + "/" for e in list_path[:-2])

    todo_vertices = [0]*mesh.v.shape[0]    
    new_vertices_u_v = [] 
    vertices_u_v = []    
    dist_vertices = [0.0]*mesh.v.shape[0] 

    my_normals = np.array(compute_normals.compute_normals(mesh.v.copy().astype(float).tolist(), mesh.f.copy().astype(int).tolist()))

    for v in range(mesh.v.shape[0]):
        vtx1 = (((mesh.v[v,0] + cam['t'][0])*cam['f'])/(mesh.v[v,2] + cam['t'][2]) + image.shape[1]/2.0) 
        vty1 = (((mesh.v[v,1] + cam['t'][1])*cam['f'])/(mesh.v[v,2] + cam['t'][2]) + image.shape[0]/2.0)
        new_vertices_u_v.append([vtx1,vty1,mesh.v[v,2]])
        vertices_u_v.append([vtx1,vty1,mesh.v[v,2]])       
        

    image_label = cv2.imread(path + label_folder + list_path[-1][:-4] + ".png",0)
    
    to_draw = np.ones(image.shape,dtype=np.uint8)

    for i in range(14):

        label = np.where(image_label == label_map_cnn[i+1],np.ones(image_label.shape,dtype=np.uint8)*255,np.zeros(image_label.shape,dtype=np.uint8))
        contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        to_draw = np.zeros(image.shape,dtype=np.uint8)
        cv2.drawContours(to_draw, contours, -1, label_colours[i], 3)
        #cv2.imshow('ImageWindow', label)
        #cv2.waitKey()
        for s,c in enumerate(contours):
            for p,point in enumerate(c):                 
                 dist = limit_deformation[i+1]
                 id_close = -1
                 for v in range(mesh.v.shape[0]):
                     if ((int(pose_partes[v]) == (i +1))): 
                         d = distancia(vertices_u_v[v],[point[0][0],point[0][1],vertices_u_v[v][2]])
                         d = (d*vertices_u_v[v][2])/cam['f']
                         if d < dist:
                            id_close = v
                            dist = d
                 
                 if id_close > -1 and intersect_is_valid(s,p,contours,vertices_u_v[id_close],point[0]):
                     if dist > dist_vertices[id_close]:
                         dist_vertices[id_close] = dist
                         todo_vertices[id_close] = 1
                         new_vertices_u_v[id_close][0] = point[0][0]
                         new_vertices_u_v[id_close][1] = point[0][1]
                          #cv2.line(image, (int(new_vertices_u_v[id_close][0]), int(new_vertices_u_v[id_close][1])), (int(vertices_u_v[id_close][0]), int(vertices_u_v[id_close][1])), label_colours[i], 2)
    
        #cv2.imshow('ImageWindow', image)
        #cv2.waitKey()
    


  
    new_vertices = mesh.v.copy().astype(float).tolist()
    
    label_map_cnn_color = [1, 6, 19, 1, 12, 13, 11, 5, 4, 7, 10, 14, 13, 3, 8]


    #image = np.ones_like(image)*255

    
    for v in range(mesh.v.shape[0]):
         if todo_vertices[v] == 1:
              new_vertices[v][0] = ((new_vertices_u_v[v][0] - image.shape[1]/2.0)*(mesh.v[v,2] + cam['t'][2]))/cam['f'] - cam['t'][0]          
              new_vertices[v][1] = ((new_vertices_u_v[v][1] - image.shape[0]/2.0)*(mesh.v[v,2] + cam['t'][2]))/cam['f'] - cam['t'][1]
              lineThickness = 2
              #cv2.line(image, (int(new_vertices_u_v[v][0]), int(new_vertices_u_v[v][1])), (int(vertices_u_v[v][0]), int(vertices_u_v[v][1])), label_colours[label_map_cnn_color[int(pose_partes[v])]], lineThickness)
           
    #cv2.imwrite(filename + "deform.jpg", image)
    
   
    
        


    return mesh.v.copy().astype(float).tolist(), new_vertices,mesh.f.copy().astype(int).tolist(),todo_vertices,inv_trasformation  

def deform_mesh(filename,model,cam,label_folder):
    
    vertices,new_vertices,faces0,todo_vertices,inv_trasformation = get_mesh_to_deformation(filename,model,cam,label_folder)       
    deformed = deformation.deformation(vertices,todo_vertices,new_vertices,faces0,50)

    t_vertices = np.array(deformed) - np.array(cam['trans']).reshape((1,3))
         
    rest_t_vertices = np.vstack((t_vertices.T, np.ones((1, t_vertices.shape[0]))))


    v_t_w = (inv_trasformation[:,0,:] * rest_t_vertices[0, :].reshape((1, -1)) + 
             inv_trasformation[:,1,:] * rest_t_vertices[1, :].reshape((1, -1)) + 
             inv_trasformation[:,2,:] * rest_t_vertices[2, :].reshape((1, -1)) + 
             inv_trasformation[:,3,:] * rest_t_vertices[3, :].reshape((1, -1))).T
 
    v_t = v_t_w[:,:-1]/v_t_w[:,[3,3,3]] 

    return (v_t)



def render(filename,model, image, cam, steps=1,segmented=False, scale=1.,crop=False,cropinfo=[]):
    """Render a sequence of views from a fitted body model."""
    assert steps >= 1

    if segmented:
        texture = filename + '_tex.png'
    else:
        texture = None

    print filename
    mesh = copy(_TEMPLATE_MESH)
    #cmesh = Mesh(_os.path.join(_os.path.dirname(__file__),
    #                           'template-bodyparts-corrected-labeled-split5.ply'))
    #mesh.vc = cmesh.vc.copy()
    mesh.vc = _COLORS['pink']

    # render ply
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']
    
    model.trans[:] = cam['trans']
   

    mesh.v = model.r


    w, h = (image.shape[1], image.shape[0])
    dist = np.abs(cam['t'][2] - np.mean(mesh.v, axis=0)[2])
    

    print 'Vai criar'

    if crop:
         scale = 2.0
         rn = _create_renderer(w=int(cropinfo[1]*scale),
                          h=int(cropinfo[0]*scale),
                          near=1.,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          #c=np.array(cam['cam_c']),
                          texture=texture)
    else:
        rn = _create_renderer(w=int(w*scale),
                          h=int(h*scale),
                          near=1.,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          #c=np.array(cam['cam_c']),
                          texture=texture)

    print 'Criou'

    light_yrot = np.radians(120)
    base_mesh = copy(mesh)
    renderings = []
    
    for angle in np.linspace(0., 2. * np.pi, num=steps, endpoint=False):
        imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture)
        renderings.append(imtmp * 255.)
    return renderings


def render_mesh(filename,model, image, cam,v_t,texture, write_mesh=False,out_name_mesh=None,steps=1,segmented=False, scale=1.,crop=False,cropinfo=[]):
    """Render a sequence of views from a fitted body model."""
    assert steps >= 1
 
    mesh = copy(_TEMPLATE_MESH)
    #cmesh = Mesh(_os.path.join(_os.path.dirname(__file__),
    #                           'template-bodyparts-corrected-labeled-split5.ply'))
    #mesh.vc = cmesh.vc.copy()
    mesh.vc = _COLORS['pink']
   
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']      
    model.trans[:] = cam['trans'] 
    #mesh.v = model.r

    mesh.v = v_t

    #pdb.set_trace()

    mesh.v =  np.array(_verts_core(model.pose, mesh.v, model.J, model.weights, model.kintree_table)) + np.array(cam['trans']).reshape((1,3))

    #pdb.set_trace()

    #new_v_t =  _verts_core(model.pose, v_t, model.J, model.weights, model.kintree_table)
     
    
    #mesh.v = mesh.v + np.array(new_v_t) 
    w, h = (image.shape[1], image.shape[0])
    dist = np.abs(cam['t'][2] - np.mean(mesh.v, axis=0)[2])
    

    print 'Vai criar'

    if crop:
         scale = 2.0
         rn = _create_renderer_mesh(w=int(cropinfo[1]*scale),
                          h=int(cropinfo[0]*scale),
                          near=1.,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          #c=np.array(cam['cam_c']),
                          texture=texture)
    else:
        rn = _create_renderer_mesh(w=int(w*scale),
                          h=int(h*scale),
                          near=1.,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          #c=np.array(cam['cam_c']),
                          texture=texture)

    print 'Criou'

    light_yrot = np.radians(120)

    #mesh.write_obj(filename + "_deformed.obj")
    
    base_mesh = copy(mesh)   

    
    renderings = []
    
    for angle in np.linspace(0., 2. * np.pi, num=steps, endpoint=False):
        if write_mesh:
            print ("Flag on")
            imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture,out_mesh_name= out_name_mesh + "_mesh.obj",out_texture_name= out_name_mesh + "_tex.jpg")
        else:
            print ("Flag off")
            imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture)


        renderings.append(imtmp * 255.)
    return renderings

@_click.command()
@_click.argument('filename_motion', type=_click.Path(exists=True, readable=True))
@_click.argument('filename_source', type=_click.Path(exists=True, readable=True))
@_click.argument('out_path', type=_click.Path(exists=True, readable=True))
@_click.option('--background_folder',
               type=_click.STRING,
               help='Relative Path to backgroud folder',
               default="./background/")
@_click.option('--pose_folder',
               type=_click.STRING,
               help='Relative Path to pose folder',
               default="./smpl_pose/")

@_click.option('--pose_folder_target',
               type=_click.STRING,
               help='Relative Path to pose folder',
               default="./smpl_pose/")

@_click.option('--img_folder',
               type=_click.STRING,
               help='Relative Path to img folder',
               default="./images/")

@_click.option('--seg_folder',
               type=_click.STRING,
               help='Relative Path to seg soft folder',
               default="./segmentations/")

@_click.option('--pkl_end',
               type=_click.STRING,
               help='retargeting target pkl ends',
               default="ret.pkl")

@_click.option('--label_folder',
               type=_click.STRING,
               help='Relative Path to label parts folder',
               default="./semantic_label/")

@_click.option('--model_type',
               type=_click.INT,
               help='neutral(0),male(1),feme(2)',
               default=0)

@_click.option('--write_mesh', is_flag=True)

@_click.option('--use_bg_scale', is_flag=True)

def cli(filename_motion,filename_source,out_path,background_folder="./background/",pose_folder="./smpl_pose/",pose_folder_target="./smpl_pose/",img_folder="./images/",seg_folder="./segmentations/",pkl_end="ret.pkl",label_folder="./semantic_label/", model_type=0,write_mesh=False,use_bg_scale=False):
    """Render a 3D model for an estimated body fit. Provide the image (!!) filename."""

    if model_type == 1:
        model = _MODEL_M
    elif model_type == 2: 
        model = _MODEL_F 
    else: 
        model = _MODEL_N 

    if _os.path.isdir(filename_motion):
        processing_folder = True
        folder_name = filename_motion[:]
        images_bg_motion = natsorted(_glob.glob(_os.path.join(folder_name + background_folder, '*')))
        images_motion = sorted(_glob.glob(_os.path.join(folder_name + img_folder, '*')))
        images_motion = [im for im in images_motion if not (im.endswith('tex.png') or im.endswith('.txt'))] 
        pose_motion = sorted(_glob.glob(_os.path.join(folder_name + pose_folder_target, '*' + pkl_end)))     

        #images = [im + '.npy.vis.png' for im in images]
    else:
        print "error filename_motion"
        exit()
        
    
    if _os.path.isdir(filename_source):
        folder_name2 = filename_source[:] 
        images_source = sorted(_glob.glob(_os.path.join(folder_name2 + img_folder, '*')))
        images_source = [im for im in images_source if not (im.endswith('tex.png') or im.endswith('.txt'))] 
        pose_source = sorted(_glob.glob(_os.path.join(folder_name2 + pose_folder, '*.pkl')))

        #images = [im + '.npy.vis.png' for im in images]
    else:
        print "error filename_source"
        exit()


    print "Start defomation of the source"

    Deformed_vertices = []
    
    Deformed_cp = []
 
    Deformed_tex = []

    Deformed_seg = []

    

    for index,image_s in enumerate(images_source):
    	#if index != 6:
        # 	continue
    	print "OK"
        filename_source = image_s
        print filename_source 
                         
        image = cv2.imread(filename_source,-1)
        
        if image.shape[2] < 4:
            list_path_my_dir = filename_source.split("/")
    	    path_parent = ''.join(e + "/" for e in list_path_my_dir[:-2])
            image_label = cv2.imread(path_parent + seg_folder + list_path_my_dir[-1][:-4] + ".png",0)
            image_label_2 = cv2.imread(path_parent + label_folder + list_path_my_dir[-1][:-4] + ".png",0)


            #foreground = np.where(image_label > 50,np.ones(image_label.shape,dtype=np.uint8)*255,np.zeros(image_label.shape,dtype=np.uint8))
            #erosion_size = 7
            #element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
            #foreground = cv2.erode(foreground , element)
            image = np.stack((image[:,:,0],image[:,:,1],image[:,:,2],image_label,image_label_2),axis=-1)


        with open(pose_source[index]) as f:
            cam = pickle.load(f)

        
        if index == 0:
            target_shape = cam['betas']

        v_t = deform_mesh(filename_source,model,cam,label_folder) 

        Deformed_vertices.append(v_t)  

        my_tex,cp = texture_vis(model, image, cam,v_t,image)


        Deformed_seg.append(my_tex[:,:,3])




        if my_tex.shape[2] > 4:
            #out_fname = image_s + '_vis_0.jpg'
            #print out_fname
            #cv2.imwrite(out_fname, my_tex[:,:,:3])

            #my_tex[:,:,0] = np.where(my_tex[:,:,3] > 250 , my_tex[:,:,0], np.zeros_like(my_tex[:,:,0]))
            #my_tex[:,:,1] = np.where(my_tex[:,:,3] > 250 , my_tex[:,:,1], np.zeros_like(my_tex[:,:,1]))
            #my_tex[:,:,2] = np.where(my_tex[:,:,3] > 250 , my_tex[:,:,2], np.zeros_like(my_tex[:,:,2]))

            #out_fname = image_s + '_vis_1.jpg'
            #print out_fname
            #cv2.imwrite(out_fname, my_tex[:,:,:3])

            my_tex[:,:,0] = np.where(my_tex[:,:,4] == tex_map_partes_cnn[:,:] ,my_tex[:,:,0], np.zeros_like(my_tex[:,:,0]))
            my_tex[:,:,1] = np.where(my_tex[:,:,4] == tex_map_partes_cnn[:,:], my_tex[:,:,1], np.zeros_like(my_tex[:,:,1]))
            my_tex[:,:,2] = np.where(my_tex[:,:,4] == tex_map_partes_cnn[:,:], my_tex[:,:,2], np.zeros_like(my_tex[:,:,2]))

            my_tex[:,:,0] = np.where(tex_A_pose > 10 ,my_tex[:,:,0], np.zeros_like(my_tex[:,:,0]))
            my_tex[:,:,1] = np.where(tex_A_pose > 10 , my_tex[:,:,1], np.zeros_like(my_tex[:,:,1]))
            my_tex[:,:,2] = np.where(tex_A_pose > 10 , my_tex[:,:,2], np.zeros_like(my_tex[:,:,2]))



            #out_fname = image_s + '_tex.jpg'
            #print out_fname
            #cv2.imwrite(out_fname, my_tex[:,:,:3])
            
  

        Deformed_tex.append(my_tex[:,:,:3]) 

        '''renderings = render_mesh(filename_source,model, image, cam,v_t,my_tex[:,:,:3])
        renderings2 = render_mesh(filename_source,model, image, cam,v_t,None)
        renderings1 = render(filename_source,model, image, cam)

        for ridx, rim in enumerate(renderings):
            out_fname = out_path + "/" + image_s.split("/")[-1]
            print out_fname
            cv2.imwrite(out_fname + '_vis_source.jpg', rim)

            cv2.imwrite(out_fname + '_vis_source_pose.jpg', renderings1[ridx])
            cv2.imwrite(out_fname + '_vis_source_pose_deformated.jpg', renderings2[ridx])
            cv2.imwrite(out_fname + '_tex_source.jpg',my_tex[:,:,:3])'''

       
     

        Deformed_cp.append(cp)  

        
              

    
    my_tex_median = np.stack(Deformed_tex, axis=3)

    #for i in range(my_tex_median.shape[3]):
          
    

    m = np.ma.masked_less(my_tex_median, 2)

    tex_final = np.ma.median(m, axis=3)
     
    #pdb.set_trace()

    mask_0 = cv2.cvtColor(tex_final.astype(np.uint8), cv2.COLOR_BGR2GRAY) 
    mask = np.where(mask_0 < 3,np.ones_like(mask_0)*255,np.zeros_like(mask_0))  

    #dilation_size = 10
    #element = cv2.getStructuringElement(cv2.MORPH_RECT,(2*dilation_size + 1 , 2*dilation_size + 1), (dilation_size,dilation_size))
    #mask = cv2.dilate(mask,element)   
    
    tex_not_fill = np.stack((tex_final[:,:,0],tex_final[:,:,1],tex_final[:,:,2],255*np.ones_like(mask)- mask),axis=2) 

       
    tex_not_fill_remap = texture_remap(tex_not_fill)

    #cv2.imwrite(_os.path.abspath(out_path + "tex_saida_0_tmp.png"),tex_not_fill_remap)
    
    cv2.imwrite(_os.path.abspath(out_path + "tex_saida_0.png"),tex_not_fill_remap )

    print ("Doing Inpaint")

    subprocess.call(['./run_inpaint.sh', _os.path.abspath(out_path + 'tex_saida_0.png'), _os.path.abspath(out_path + 'tex_saida_0_fill.png')])  
    
 
    tex_final = texture_remap(cv2.imread(_os.path.abspath(out_path +  "tex_saida_0_fill.png")),direction=False)

    #tex_final = cv2.inpaint(tex_not_fill_remap[:,:,:3].astype(np.uint8),tex_not_fill_remap[:,:,3].astype(np.uint8),9,cv2.INPAINT_NS)

    tex_final = tex_final*(mask[:,:,np.newaxis]/255) + tex_not_fill[:,:,:3]*(np.ones_like(mask[:,:,np.newaxis]) - mask[:,:,np.newaxis]/255)

     

    mask_0 = cv2.cvtColor(tex_final.astype(np.uint8), cv2.COLOR_BGR2GRAY) 

    mask = np.where(mask_0 < 1,np.ones_like(mask_0)*255,np.zeros_like(mask_0))

    #pdb.set_trace()

   
    tex_final = cv2.inpaint(tex_final.astype(np.uint8),mask,3,cv2.INPAINT_NS)

    #cv2.imwrite(path_parent + "tex_saida_1.png",tex_final)

    print ("Done")

   
    cv2.imwrite(out_path + "tex_saida_1.jpg", tex_final)
        
    '''for index,image_s in enumerate(images_source):
        mask_0 = cv2.cvtColor(Deformed_tex[index].astype(np.uint8), cv2.COLOR_BGR2GRAY) 
        mask = np.where(mask_0 > 3,np.ones(mask_0.shape),np.zeros(mask_0.shape)) 
        mask = np.stack((mask,mask,mask),axis=2)
        tex_to_use =  tex_final*(np.ones(tex_final.shape) - mask ) + mask*Deformed_tex[index]
        Deformed_tex[index] = tex_to_use
        

    for index,image_s in enumerate(images_source):
        cv2.imshow('ImageWindow', Deformed_tex[index]/255)
        cv2.waitKey()'''
    
    
       

   
    
    print "End deformation of the source"

    print "Star deformation by vision"

    vt_old = np.zeros((6890,3))

    for index,image_m in enumerate(images_motion):
        print image_m
        image = cv2.imread(image_m)

        with open(pose_motion[index]) as f:
            cam = pickle.load(f)

        cam['betas'] = target_shape

        #cam['trans'][1] = cam['trans'][1] - 0.08

        tex_m,cp_m = texture_vis(model, image, cam,model.v_posed,image)

        out_fname = out_path + "/" + image_m.split("/")[-1]
        #cv2.imwrite(out_fname + '_tex_vis.jpg',tex_m)


        mask_0_m = cv2.cvtColor(tex_m.astype(np.uint8), cv2.COLOR_BGR2GRAY) 
        mask_m = np.where(mask_0_m > 3,np.ones_like(mask_0_m)*255,np.zeros_like(mask_0_m)) 

        select_ids = [0]*14 
        select_soma = [10000000]*14
        
        for index_2,image_s in enumerate(images_source):
             mask_0 = cv2.cvtColor(Deformed_tex[index_2].astype(np.uint8), cv2.COLOR_BGR2GRAY) 
             mask = np.where(mask_0 > 3,np.ones_like(mask_0)*255,np.zeros_like(mask_0)) 
             tex_now = cv2.bitwise_xor(mask,mask_m)             
             soma_now = [0.0]*14
             for s in range(len(soma_now)):
                 soma_now[s] = cv2.countNonZero(tex_now*np.where(tex_map_partes == (s + 1)*15,np.ones(tex_map_partes.shape),np.zeros(tex_map_partes.shape)))


             #print soma_now

             for i in range(len(select_soma)):
                 if select_soma[i] > soma_now[i]:
                     select_soma[i] = soma_now[i]
                     select_ids[i] = index_2
             #print select_soma
             #print select_ids

        print select_ids

        vt_s_m = np.array(model.v_posed)

        for v in range(vt_s_m.shape[0]):
            if Deformed_cp[select_ids[int(pose_partes[v]) -1]][v,0] > 0.5:
               vt_s_m[v,:] = Deformed_vertices[select_ids[int(pose_partes[v]) -1]][v,:]

        if index > 0 :
            vt_s_m = (vt_s_m + vt_old)/2.0
           
        vt_old = vt_s_m 

        tex_now_final = np.zeros_like(tex_final)
        tex_now_final_debug = np.zeros_like(tex_final)


        for i in range(tex_final.shape[0]):
            for j in range(tex_final.shape[1]):
                if (float(Deformed_tex[select_ids[int(tex_map_partes[i,j]/15 -1)]][i,j,0]) + float(Deformed_tex[select_ids[int(tex_map_partes[i,j]/15 -1)]][i,j,1]) + float(Deformed_tex[select_ids[int(tex_map_partes[i,j]/15 -1)]][i,j,2])) > 9:
                    tex_now_final[i,j,:] = Deformed_tex[select_ids[int(tex_map_partes[i,j]/15 -1)]][i,j,:]
                    tex_now_final_debug[i,j,:] = Deformed_tex[select_ids[int(tex_map_partes[i,j]/15 -1)]][i,j,:]
                else:
                    tex_now_final[i,j,:] = tex_final[i,j,:]  

   
        if  index > len(images_bg_motion) - 1:
            if len(images_bg_motion) == 0:
               back = np.ones_like(image)*255

            else:
               back = cv2.imread(images_bg_motion[len(images_bg_motion) -1])
        else:
            back = cv2.imread(images_bg_motion[index])

        if use_bg_scale:
            renderings = render_mesh(image_m,model,image, cam,vt_s_m,tex_now_final,write_mesh=write_mesh,out_name_mesh= out_path + "/" + image_m.split("/")[-1],crop=True,cropinfo=back.shape)
        else:
            renderings = render_mesh(image_m,model,image, cam,vt_s_m,tex_now_final,write_mesh=write_mesh,out_name_mesh= out_path + "/" + image_m.split("/")[-1])

        renderings[0] = renderings[0].astype(np.uint8)
        mask = cv2.cvtColor(renderings[0],cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(mask,254,1.0,cv2.THRESH_BINARY)

        image_seg = feather_blending(renderings[0],cv2.resize(back,(renderings[0].shape[1],renderings[0].shape[0])),(np.ones_like(mask.astype(np.uint8)) - mask.astype(np.uint8))*255,edge=7)
        #pdb.set_trace()

        #image_seg = (back*mask[:,:,np.newaxis] +  renderings[0]*(np.ones((renderings[0].shape[0],renderings[0].shape[1],3)) - mask[:,:,np.newaxis]))

        #out_fname = out_path + "/" + image_m.split("/")[-1] + '_text.png'
        #cv2.imwrite(out_fname, tex_now_final)

        out_fname = out_path + "/" + image_m.split("/")[-1] + '_vis.jpg'
        cv2.imwrite(out_fname, image_seg)
       
       

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _logging.getLogger("opendr.lighting").setLevel(_logging.FATAL)  # pylint: disable=no-attribute
    cli()  # pylint: disable=no-value-for-parameter
