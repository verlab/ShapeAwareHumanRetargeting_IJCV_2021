%module deformation
%{
#include "deformation.h"
%}
%include "std_vector.i"
namespace std {

  /* On a side note, the names VecDouble and VecVecdouble can be changed, but the order of first the inner vector matters! */
  %template(VecDouble) vector<double>;
  %template(VecInt) vector<int>;
  %template(VeVeccInt) vector< vector<int> >;
  %template(VecVecdouble) vector< vector<double> >;
}

%include "deformation.h"
