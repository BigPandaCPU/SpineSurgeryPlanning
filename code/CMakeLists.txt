cmake_minimum_required(VERSION 3.12)

project(SpinePointCloudSeg)

set(CMAKE_CXX_STANDARD 11)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "D:/software/onnxruntime-win-x64-1.9.0")
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

# 添加可执行文件
add_executable(SpineSeg "src/loadonnx.cpp" "src/main.cpp")

# 将以.dll结尾的文件复制到生成的debug目录下
 file(GLOB DLLS_FILES "onnxruntime/lib/*.dll")
 file(COPY ${DLLS_FILES} DESTINATION ${CMAKE_BINARY_DIR})

# 将以.onnx结尾的文件复制到生成的debug目录下
file(GLOB ONNX_FILES "src/*.onnx")
file(COPY ${ONNX_FILES} DESTINATION ${CMAKE_BINARY_DIR})
