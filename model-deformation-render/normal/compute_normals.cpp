#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <iostream>
#include <fstream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;
typedef K::Vector_3 Vector;
typedef CGAL::Surface_mesh<Point> Surface_mesh;
typedef boost::graph_traits<Surface_mesh>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Surface_mesh>::face_descriptor   face_descriptor;

using namespace std;

std::vector< std::vector<double> > compute_normals(std::vector< std::vector<double> > my_vertices, std::vector< std::vector<int> > my_faces){


  vector <Surface_mesh::Vertex_index> my_vertor_id;  
  Surface_mesh my_mesh;

  for (int v = 0; v < my_vertices.size(); v++){ 
       Surface_mesh::Vertex_index u = my_mesh.add_vertex(Point(my_vertices[v][0],my_vertices[v][1],my_vertices[v][2]));
       my_vertor_id.push_back(u);  
  }
  
  for (int f = 0; f < my_faces.size(); f++){ 
       my_mesh.add_face(my_vertor_id[my_faces[f][0]], my_vertor_id[my_faces[f][1]], my_vertor_id[my_faces[f][2]]);  
  }


  Surface_mesh::Property_map<face_descriptor, Vector> fnormals =
    my_mesh.add_property_map<face_descriptor, Vector>
      ("f:normals", CGAL::NULL_VECTOR).first;
  Surface_mesh::Property_map<vertex_descriptor, Vector> vnormals =
    my_mesh.add_property_map<vertex_descriptor, Vector>
      ("v:normals", CGAL::NULL_VECTOR).first;

  CGAL::Polygon_mesh_processing::compute_normals(my_mesh,
        vnormals,
        fnormals,
        CGAL::Polygon_mesh_processing::parameters::vertex_point_map(my_mesh.points()).
        geom_traits(K()));

  /*std::cout << "Face normals :" << std::endl;

  BOOST_FOREACH(face_descriptor fd, faces(my_mesh)){
    std::cout << fnormals[fd] << std::endl;
  }
  std::cout << "Vertex normals :" << std::endl;*/
  
  std::vector< std::vector<double> >  normais;

  BOOST_FOREACH(vertex_descriptor vd, vertices(my_mesh)){
    std::vector<double> vert;
    vert.push_back(vnormals[vd][0]);
    vert.push_back(vnormals[vd][1]);
    vert.push_back(vnormals[vd][2]);
    normais.push_back(vert);

  }
  
  
  
  return normais;
}

