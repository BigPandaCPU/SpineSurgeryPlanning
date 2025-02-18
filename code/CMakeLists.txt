cmake_minimum_required(VERSION 3.12)

project(SpinePointCloudSeg)

set(CMAKE_CXX_STANDARD 11)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "D:/software/onnxruntime-win-x64-gpu-1.9.0")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# 添加ONNX Runtime库头文件的包含路径
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/include")
# 添加ONNX Runtime库的链接路径
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
# 添加ONNX Runtime库的链接库
LINK_LIBRARIES("${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")

# 添加open3D库
set(Open3D_DIR "D:/software/open3d-devel-windows-amd64-0.16.0/CMake")
find_package(Open3D REQUIRED)

find_package(VTK REQUIRED)

# 添加可执行文件
file(GLOB FILE_HEADER "src/*.h")
file(GLOB FILE_SOURCE "src/*.cpp")

include(${VTK_USE_FILE})

add_executable(SpineSeg ${FILE_HEADER} ${FILE_SOURCE})
target_link_libraries(SpineSeg Open3D::Open3D ${VTK_LIBRARIES})

# 将以.dll结尾的文件复制到生成的debug目录下
file(GLOB DLLS_FILES "onnxruntime/lib/*.dll")
file(COPY ${DLLS_FILES} DESTINATION ${CMAKE_BINARY_DIR})

# 将以.onnx结尾的文件复制到生成的debug目录下
file(GLOB ONNX_FILES "src/*.onnx")
file(COPY ${ONNX_FILES} DESTINATION ${CMAKE_BINARY_DIR})
