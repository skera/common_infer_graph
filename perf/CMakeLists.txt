cmake_minimum_required(VERSION 3.10)
project(TrtUnpackInference)

set(CMAKE_CXX_STANDARD 17)

# 设置 TensorRT 安装路径
set(TENSORRT_ROOT "/data/app/TensorRT-10.9.0.34")

# 查找 TensorRT 头文件和库文件
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT}/include)
find_library(TRT_INFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(TRT_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)
find_library(TRT_ONNX_PARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 包含头文件目录
include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})

# 添加可执行文件
add_executable(trt_perf tensorrt.cpp)

# 链接所需库
target_link_libraries(trt_perf
    ${TRT_INFER_LIB}
    ${TRT_PLUGIN_LIB}
    ${TRT_ONNX_PARSER_LIB}
    ${CUDA_LIBRARIES}
    dl
    pthread
)
