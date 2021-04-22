#ifndef _deformation
#define _deformation

#include <vector>

std::vector< std::vector<double> > deformation (std::vector< std::vector<double> > my_vertices, std::vector<int> todo_vertices,std::vector< std::vector<double> > my_new_vertices, std::vector< std::vector<int> > my_faces,int deform_iterations);

#endif
