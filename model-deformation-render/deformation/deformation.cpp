#include <vector>
#include "deformation.h"
#include <fstream>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh_deformation.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3>                  Mesh;

typedef boost::graph_traits<Mesh>::vertex_descriptor    vertex_descriptor;
typedef boost::graph_traits<Mesh>::vertex_iterator        vertex_iterator;
typedef boost::graph_traits<Mesh>::halfedge_iterator    halfedge_iterator;

typedef CGAL::Surface_mesh_deformation<Mesh> Surface_mesh_deformation;


using namespace std;

std::vector< std::vector<double> > deformation (std::vector< std::vector<double> > my_vertices, std::vector<int> todo_vertices,std::vector< std::vector<double> > my_new_vertices, std::vector< std::vector<int> > my_faces,int deform_iterations){

  //Create the mesh
  vector <Mesh::Vertex_index> my_vertor_id;  
  Mesh my_mesh;
  for (int v = 0; v < my_vertices.size(); v++){ 
       Mesh::Vertex_index u = my_mesh.add_vertex(Point_3(my_vertices[v][0],my_vertices[v][1],my_vertices[v][2]));
       my_vertor_id.push_back(u);  
  }
  
  for (int f = 0; f < my_faces.size(); f++){ 
       my_mesh.add_face(my_vertor_id[my_faces[f][0]], my_vertor_id[my_faces[f][1]], my_vertor_id[my_faces[f][2]]);  
  }

  /*cout << "My vertices:" << endl;
  for (int i = 0; i < my_vertor_id.size();i++)
       cout << i << ": " << my_mesh.point(my_vertor_id[i]) << " " << endl;*/

  // Create a deformation object
  Surface_mesh_deformation deform_mesh(my_mesh);

  // Definition of the region of interest (use the whole mesh)
  vertex_iterator vb,ve;
  boost::tie(vb, ve) = vertices(my_mesh);
  deform_mesh.insert_roi_vertices(vb, ve);

  //Definition of the control points

  std::vector<vertex_descriptor > control_points;
  std::vector<int> control_points_id;

  for(int i = 0; i < todo_vertices.size();i++){
       if(todo_vertices[i] == 0)
           continue;
       else{
           vertex_descriptor control_1 = *std::next(vb, i); 
           deform_mesh.insert_control_vertex(control_1);
           control_points.push_back(control_1);
           control_points_id.push_back(i);
       }

  }
  
  
  // teste erros
 
  bool is_matrix_factorization_OK = deform_mesh.preprocess();
  if(!is_matrix_factorization_OK){
    std::cerr << "Error in preprocessing, check documentation of preprocess()" << std::endl;
    std::vector< std::vector<double> >  empty_list;
    return empty_list;
  }
    
  cout << control_points_id.size() << endl;

  for(int i = 0; i < control_points_id.size();i++){
      Surface_mesh_deformation::Point constrained_pos(my_new_vertices[control_points_id[i]][0], my_new_vertices[control_points_id[i]][1], my_new_vertices[control_points_id[i]][2]);
      deform_mesh.set_target_position(control_points[i], constrained_pos);     

  }

  cout << "Start deformation" << endl;

  //deform_mesh.deform(deform_iterations,0.0);
  deform_mesh.deform();

  cout << "End deformation" << endl;
     
  std::vector< std::vector<double> >  deformeted_vertices;

  for (int i = 0; i < my_vertor_id.size();i++){
       std::vector<double> vert;
       vert.push_back(my_mesh.point(my_vertor_id[i])[0]);
       vert.push_back(my_mesh.point(my_vertor_id[i])[1]);
       vert.push_back(my_mesh.point(my_vertor_id[i])[2]);
       deformeted_vertices.push_back(vert);
  }
  
  return deformeted_vertices;


}
