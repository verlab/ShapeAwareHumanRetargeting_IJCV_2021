#!/usr/bin/env python
"""Render a model."""
# pylint: disable=invalid-name
import sys as _sys
import os as _os
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
_sys.path.append('../')

import pdb

from up_tools.mesh import Mesh
from up_tools.camera import rotateY,translateG
_sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
from config import SMPL_FP
_sys.path.insert(0, SMPL_FP)

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

_MOD_PATH = _os.path.abspath(_os.path.dirname(__file__))
# Models:
# Models:

_MODEL_NEUTRAL_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'lbs_tj10smooth6_0fixed_normalized_locked_hybrid_model_novt.pkl')
_MODEL_MALE_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

_MODEL_FEME_FNAME = _os.path.join(
    _MOD_PATH, '.', 'models', '3D',
    'basicModel_f_lbs_10_207_0_v1.0.0.pkl')

_MODEL_N = _load_model(_MODEL_NEUTRAL_FNAME)
_MODEL_M = _load_model(_MODEL_MALE_FNAME)
_MODEL_F = _load_model(_MODEL_FEME_FNAME)

# Torso joint IDs (used for estimating camera position).

_REGRESSORS_FNAME_M = _os.path.join(
    _MOD_PATH, '.', 'models', '3D', 'regressors_locked_normalized_male.npz')

_REGRESSORS_FNAME_F = _os.path.join(
    _MOD_PATH, '.', 'models', '3D', 'regressors_locked_normalized_female.npz')

_REGRESSORS_FNAME_N = _os.path.join(
    _MOD_PATH, '.', 'models', '3D', 'regressors_locked_normalized_hybrid.npz')


_REGRESSORS_N = np.load(_REGRESSORS_FNAME_N)
_REGRESSORS_M = np.load(_REGRESSORS_FNAME_M)
_REGRESSORS_F = np.load(_REGRESSORS_FNAME_F)
_TEX_MAP_FNAME = _os.path.join(
    _os.path.dirname(__file__), '..', 'models', 'mapeamento2.obj')

_COLORS = {
    'pink': [.6, .6, .8],
    'cyan': [.7, .75, .5],
    'yellow': [.5, .7, .75],
    'grey': [.7, .7, .7],
}

def Template_tex():
	file = open(_TEX_MAP_FNAME,'r')
	lines = file.readlines()
	vt = []
	f = []
	for l in lines:
		l.strip()
		if l[0] == 'v' and l[1] == 't':	
			valores = l.split()
			vt.append([float(i) for i in valores[1::]])
		elif l[0] == 'f':
			valores = l.split()                
			f.append([(int(g) - 1) for g in valores[1::]])
	return np.array(vt),np.array(f)


def _create_renderer(  # pylint: disable=too-many-arguments
        w=640,
        h=480,
        rt=np.zeros(3),
        t=np.zeros(3),
        f=None,
        c=None,
        k=None,
        near=0.1,
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

    #print f,c,k,rt,t

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}
    if texture is not None:
        rn.texture_image = np.asarray(cv2.imread(texture), np.float64)/255.
        rn.texture_image = rn.texture_image[:,:,::-1]


        print rn.texture_image.shape
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


def _simple_renderer(rn, meshes, yrot=0, texture=None):
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
def render(filename,model, image, cam, segmented=False, scale=1.):
    """Render a sequence of views from a fitted body model."""
    


    if segmented:
        texture = filename + '_tex.png'

    else:
        texture = None
    mesh = copy(_TEMPLATE_MESH)
    #cmesh = Mesh(_os.path.join(_os.path.dirname(__file__),
    #                           'template-bodyparts-corrected-labeled-split5.ply'))
    #mesh.vc = cmesh.vc.copy()
    mesh.vc = _COLORS['pink']

    # render ply
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']
     
    #cam['trans'][1] = cam['trans'][1] - 0.08
    #olhar essa translacao, error nos diferentes tipos de camera no SPIN a camera nao esta na mesma direcao ..
    
    model.trans[:] = cam['trans'] 

    #model.trans[:] = [cam['trans'][0]*-1,cam['trans'][1],cam['trans'][2]]

    print cam['trans'] 

    mesh.v = model.r
    w, h = (image.shape[1], image.shape[0])
    dist = np.abs(cam['t'][2] - np.mean(mesh.v, axis=0)[2])
    
    #pdb.set_trace()  
 
    if not('cam_c' is cam.keys()):
        cam['cam_c'] = [image.shape[1]/2,image.shape[0]/2]

    print 'Vai criar'

 
    rn = _create_renderer(w=int(w*scale),
                          h=int(h*scale),
                          near=0.1,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          c=np.array(cam['cam_c']),
                          texture=texture)

    print 'Criou'
    print cam['cam_c']

    light_yrot = np.radians(120)
    base_mesh = copy(mesh)
    renderings = []
    
    for angle in np.linspace(0., 2. * np.pi, num=1, endpoint=False):        
        mesh.v = rotateY(mesh.v, angle)
        imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture)
        renderings.append(imtmp * 255.)
    return renderings


@_click.command()
@_click.argument('filename', type=_click.Path(exists=True, readable=True))
@_click.option('--no_shape',
               type=_click.BOOL,
               is_flag=True,
               help='If set, use a segmented mesh.')

@_click.option("--scale",
               type=_click.FLOAT,
               help="Render the results at this scale.",
               default=1.)
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help='The ending to use for the images to read, if a folder is specified.',
               default='.jpg')
@_click.option('--model_type',
               type=_click.INT,
               help='neutral(0),male(1),feme(2)',
               default=1)
@_click.option('--folder_pose_suffix',
               type=_click.STRING,
               help='The ending to use for the images to read, if a folder is specified.',
               default='_ret.pkl')
@_click.option('--render_number',
               type=_click.INT,
               help='neutral(0),male(1),feme(2)',
               default=0)
def cli(filename, no_shape=False,folder_image_suffix='.jpg',scale=1.,model_type=1,folder_pose_suffix='_ret.pkl',render_number=0):
    """Render a 3D model for an estimated body fit. Provide the image (!!) filename."""
    

    if model_type == 1:
        model = _MODEL_M
    elif model_type == 2: 
        model = _MODEL_F
    else: 
        model = _MODEL_N
    
    if _os.path.isdir(filename):
        processing_folder = True
        folder_name = filename[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = sorted(_glob.glob(_os.path.join(folder_name + "/test_img/", '*' + folder_image_suffix)))
        images = [im for im in images if not (im.endswith('vis.jpg') or im.endswith('tex.png') or im.endswith('background.png') or im.endswith('foreground.png') or im.endswith('_body_0.png')) ]
        
        pose_names = sorted(_glob.glob(_os.path.join(folder_name + "/test_pose_new/", '*' + folder_pose_suffix)))

        #images = [im + '.npy.vis.png' for im in images]
    else:
        processing_folder = False
        images = [filename]
       
    

    for i,image_name in enumerate(images):
         fileending = image_name[-4:]
         pkl_fname = pose_names[i]
         crop_file = ''
         cropinfo = []
        
         _LOGGER.info("Rendering 3D model for image %s and parameters %s...",
                 image_name, pkl_fname)
         assert _os.path.exists(pkl_fname), (
        'Stored body fit does not exist for {}: {}!'.format(
            image_name, pkl_fname))

         print pkl_fname

         image = cv2.imread(image_name)
         with open(pkl_fname) as f:
             cam = pickle.load(f)

         #pdb.set_trace()

         if no_shape == True:
             cam['betas'] = np.zeros(10)
   
         renderings = render(image_name,model, image, cam,False, scale)

         for ridx, rim in enumerate(renderings):
             out_fname = pkl_fname + '_body_%d.jpg' % (render_number)
             print out_fname

             cv2.imwrite(out_fname, rim)

   
if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _logging.getLogger("opendr.lighting").setLevel(_logging.FATAL)  # pylint: disable=no-attribute
    cli()  # pylint: disable=no-value-for-parameter
