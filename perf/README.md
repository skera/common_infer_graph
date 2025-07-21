# 性能测试工具说明

该工具用于对 TensorRT 推理引擎进行性能测试和基准评估。以下是工具的主要功能和使用方法：

## 功能概述
1. **加载 TensorRT 引擎**：通过指定引擎文件路径加载推理引擎。
2. **设置输入输出 Tensor**：根据指定的 batch size 配置输入 Tensor 的形状，并分配 GPU 内存。
3. **推理执行**：进行推理操作，包括 warmup 和多次重复推理以测量性能。
4. **性能统计**：计算推理的平均时间、最小时间、最大时间以及标准差。
5. **结果输出**：打印推理结果的 Tensor 形状及部分输出值。

## 使用方法
运行命令格式如下：
```bash
./trt_perf <engine_path> <batch_size>
```
例如：
```bash
./trt_perf /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 1024
```

## 性能分析工具
可以结合 `nsys profile` 工具进行性能分析，示例如下：
```bash
nsys profile --trace=cuda,nvtx,cudnn -o <output_path> ./trt_perf <engine_path> <batch_size>
```
具体示例：
```bash
nsys profile --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_lhuc_profile_b2048_view ./trt_perf /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 2048
```

## 注意事项
- 确保提供的引擎文件路径正确。
- 根据实际需求调整 batch size。
- 使用 `nsys` 工具时，请确保已安装相关依赖。

该工具适用于多种模型的性能测试，包括 `lhuc`、`comxcom`、`mlcc`、`mmoe` 和 `ppnet` 等。  