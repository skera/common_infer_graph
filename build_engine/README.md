# README

## 概述

该项目提供了一个工具，用于将 ONNX 模型转换为 TensorRT 序列化引擎。该工具使用 C++ 实现，并利用 TensorRT 和 ONNX 解析库进行高效的模型转换和优化。

## 功能

- 解析 ONNX 模型并验证其结构。
- 支持为动态批量大小设置优化配置文件。
- 构建并序列化 TensorRT 引擎。
- 将序列化的引擎输出到指定文件。

## 使用方法

### 构建工具
使用您喜欢的 C++ 编译器编译工具，并链接 TensorRT 和 ONNX 解析库。

### 运行工具
```bash
./build_engine <onnx_path> <engine_path> <batch_min> <batch_max>
```

- `<onnx_path>`：ONNX 模型文件的路径。
- `<engine_path>`：保存序列化 TensorRT 引擎的路径。
- `<batch_min>`：优化的最小批量大小。
- `<batch_max>`：优化的最大批量大小。

### 示例命令
```bash
./build_engine /data/app/order_large/common_infer_graph/model/lhuc/lhuc.onnx /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 1 2048
./build_engine /data/app/order_large/common_infer_graph/model/comxcom/comxcom.onnx /data/app/order_large/common_infer_graph/model/comxcom/comxcom.engine 1 2048
```
