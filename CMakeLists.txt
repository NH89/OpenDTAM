# To control the setup of the module you could also set:
# the_description - text to be used as current module description
# the_label - label for current module
# OPENCV_MODULE_TYPE - STATIC|SHARED - set to force override global settings for current module
# OPENCV_MODULE_IS_PART_OF_WORLD - ON|OFF (default ON) - should the module be added to the opencv_world?
# BUILD_${the_module}_INIT - ON|OFF (default ON) - initial value for BUILD_${the_module}

# The verbose template for OpenCV module:
#
#   ocv_add_module(modname <dependencies>)

#   ocv_glob_module_sources(([EXCLUDE_OPENCL] [EXCLUDE_CUDA] <extra sources&headers>)
#                          or glob them manually and ocv_set_module_sources(...)

#   ocv_module_include_directories(<extra include directories>)

#   ocv_create_module()

#   <add extra link dependencies, compiler options, etc>

#   ocv_add_precompiled_headers(${the_module})

#   <add extra installation rules>

#   ocv_add_accuracy_tests(<extra dependencies>)

#   ocv_add_perf_tests(<extra dependencies>)

#   ocv_add_samples(<extra dependencies>)
#
#
# If module have no "extra" then you can define it in one line:
#
#   ocv_define_module(modname <dependencies>)

### NB OpenCVModule.cmake, macro(ocv_glob_module_sources), line 801 expects 
#   *.cu files to be in src/cuda/ , 
#   _and_ cuda headers to use *.hpp
#
# if(HAVE_CUDA AND exclude_cuda EQUAL -1)
#    file(GLOB lib_cuda_srcs
#         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.cu"
#    )
#    file(GLOB lib_cuda_hdrs
#         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.hpp"
#    )
#
## Where CMAKE_CURRENT_LIST_DIR: Full directory of the listfile currently being processed.

if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaOpenDTAM)      
endif()

set(the_description "CUDA-implementation of OpenDTAM")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(cudaOpenDTAM opencv_cudaimgproc opencv_cudastereo opencv_imgproc OPTIONAL opencv_cudev opencv_cudaarithm opencv_cudafilters WRAP python)
