cmake_minimum_required(VERSION 3.10)
project(BuildEngine LANGUAGES CXX)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 TensorRT 路径（也可通过环境变量 TENSORRT_ROOT 提供）
if(NOT DEFINED ENV{TENSORRT_ROOT})
  message(FATAL_ERROR "请设置环境变量 TENSORRT_ROOT 指向 TensorRT 安装目录")
endif()
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})

# 查找 TensorRT 包含文件和库
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT}/include)
find_library(TRT_INFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(TRT_PARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib)
find_library(TRT_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)

# 查找 CUDA
find_package(CUDA REQUIRED)

include_directories(${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
link_directories(${TENSORRT_ROOT}/lib)

# 可执行文件
add_executable(build_engine build_engine.cpp)

# 链接库：TensorRT core、parser、plugin、CUDA runtime，还可以加 dl（用于 dlopen）
target_link_libraries(build_engine
  ${TRT_INFER_LIB}
  ${TRT_PARSER_LIB}
  ${TRT_PLUGIN_LIB}
  ${CUDA_LIBRARIES}
  dl
)

# 如果你的插件 .so 在非标准路径，运行时需设置 LD_LIBRARY_PATH
