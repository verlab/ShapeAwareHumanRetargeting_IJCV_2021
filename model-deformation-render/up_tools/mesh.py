"""Mesh tools."""
# pylint: disable=invalid-name
import numpy as np
import random
import scipy.sparse as sp
import plyfile

class Mesh(object):  # pylint: disable=too-few-public-methods

    """An easy to use mesh interface."""

    def __init__(self, filename=None, v=None, vc=None, f=None):
        """Construct a mesh either from a file or with the provided data."""
        if filename is not None:
            assert v is None and f is None and vc is None
        else:
            assert v is not None or f is not None
            if vc is not None:
                assert len(v) == len(vc)

        if filename is not None:
            plydata = plyfile.PlyData.read(filename)
            self.v = np.hstack((np.atleast_2d(plydata['vertex']['x']).T,
                                np.atleast_2d(plydata['vertex']['y']).T,
                                np.atleast_2d(plydata['vertex']['z']).T))
            self.vc = np.hstack((np.atleast_2d(plydata['vertex']['red']).T,
                                 np.atleast_2d(plydata['vertex']['green']).T,
                                 np.atleast_2d(plydata['vertex']['blue']).T)).astype('float') / 255.
            # Unfortunately, the vertex indices for the faces are stored in an
            # object array with arrays as objects. :-/ Work around this.
            self.f = np.vstack([np.atleast_2d(elem) for
                                elem in list(plydata['face']['vertex_indices'])]).astype('uint32')
        else:
            self.v = v
            self.vc = vc
            self.f = f

    def write_ply(self, out_name):
        """Write to a .ply file."""
        vertex = rfn.merge_arrays([
            self.v.view(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),]),
            (self.vc * 255.).astype('uint8').view(
                dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
            ],
                                  flatten=True,
                                  usemask=False)
        face = self.f.view(dtype=[('vertex_indices', 'i4', (3,))])[:, 0]
        vert_el = plyfile.PlyElement.describe(vertex, 'vertex')
        face_el = plyfile.PlyElement.describe(face, 'face')
        plyfile.PlyData([
            vert_el,
            face_el
        ]).write(out_name)

    def write_obj(self, out_name):
        with open(out_name, 'w') as f:
            f.write("# OBJ file\n")
            for v in range(self.v.shape[0]):
                f.write("v " + str(self.v[v][0]) + " " + str(self.v[v][1]) + " " + str(self.v[v][2]) + "\n" )
            for v in range(self.f.shape[0]):
                f.write("f " + str(self.f[v][0] + 1) + " " + str(self.f[v][1] + 1) + " " + str(self.f[v][2] +1) + "\n" )


def write_obj(mesh, out_name):
    with open(out_name, 'w') as my_file:
        my_file.write("# OBJ file\n")
        my_file.write("mtllib " + out_name[:-3].split()[-1] + "mtl\n") 
        my_file.write("o " + out_name[:-4].split()[-1] + "\n")
        for v in range(mesh.v.shape[0]):
            my_file.write("v " + str(mesh.v[v][0]) + " " + str(mesh.v[v][1]) + " " + str(mesh.v[v][2]) + "\n" )
        for vt in range(mesh.vt.shape[0]):
            my_file.write("vt " + str(mesh.vt[vt][0]) + " " + str(mesh.vt[vt][1]) + "\n" )
       
        my_file.write("usemtl Material.001" + "\n" + "s off" + "\n")

        for f in range(mesh.f.shape[0]):
            my_file.write("f " + str(mesh.f[f][0] + 1) + "/" +  str(mesh.ft[f][0] + 1) + " "  + str(mesh.f[f][1] + 1) + "/" +  str(mesh.ft[f][1] + 1) + " "  + str(mesh.f[f][2] + 1) + "/" +  str(mesh.ft[f][2] + 1) + "\n" )












