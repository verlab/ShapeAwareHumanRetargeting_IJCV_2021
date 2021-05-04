# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


cmake_minimum_required(VERSION 3.1...3.15)
project( compute_normals )
add_compile_options(-fPIC)


find_package(CGAL QUIET)

if ( CGAL_FOUND )

  find_package(Eigen3 3.1.91) #(requires 3.2.0 or greater)
  if (EIGEN3_FOUND)
    include( ${EIGEN3_USE_FILE} )

    find_package(PythonInterp 2.7 REQUIRED)
    find_package(PythonLibs 2.7 REQUIRED)
    message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
    message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
    message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})

    add_library(compute_normals SHARED compute_normals.cpp compute_normals_wrap.cxx)
    target_link_libraries(compute_normals CGAL::CGAL)
    set_target_properties(compute_normals PROPERTIES LIBRARY_OUTPUT_NAME _compute_normals.so)
    set_target_properties(compute_normals PROPERTIES PREFIX "")
    set_target_properties(compute_normals PROPERTIES SUFFIX "")
   
    #create_single_source_cgal_program( "all_roi_assign_example_Surface_mesh.cpp" )   
    #install(TARGETS ${PROJECT_NAME} DESTINATION lib/)
  else()
    message(STATUS "NOTICE: These examples require the Eigen library, version 3.2 or later and will not be compiled.")
  endif()
else()
  
    message(STATUS "NOTICE: These exmaples require the CGAL library, and will not be compiled.")
  
endif()

