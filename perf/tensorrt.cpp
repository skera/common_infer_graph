#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cassert>
#include <chrono>
#include <cmath>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>
#include <map>

using namespace nvinfer1;
using Clock = std::chrono::high_resolution_clock;

// Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// 工具函数
size_t volume(const Dims& dims) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file: " + path);
    return {std::istreambuf_iterator<char>(file), {}};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <batch_size>\n";
        return 1;
    }

    std::string enginePath = argv[1];
    int64_t batchSize = std::stoll(argv[2]);

    // Load engine
    auto engineData = loadEngineFile(enginePath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // 构建 tensor 名到 buffer 索引映射
    std::map<std::string, int> tensorIndexMap;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
        tensorIndexMap[engine->getIOTensorName(i)] = i;

    std::vector<std::string> inputNames, outputNames;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorDataType(name) != DataType::kFLOAT) continue;
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
            inputNames.push_back(name);
        else
            outputNames.push_back(name);
    }

    // 设置 batch size for each input
    for (const auto& name : inputNames) {
        Dims dims = engine->getTensorShape(name.c_str());
        dims.d[0] = batchSize;
        context->setInputShape(name.c_str(), dims);
    }

    std::vector<void*> buffers(engine->getNbIOTensors(), nullptr);
    std::vector<std::vector<float>> hostInputs(inputNames.size()), hostOutputs(outputNames.size());

    // 输入 buffer
    for (size_t i = 0; i < inputNames.size(); ++i) {
        const std::string& name = inputNames[i];
        Dims dims = context->getTensorShape(name.c_str());
        size_t size = volume(dims);
        void* dptr;
        cudaMalloc(&dptr, size * sizeof(float));
        buffers[tensorIndexMap[name]] = dptr;
        context->setTensorAddress(name.c_str(), dptr);
        hostInputs[i].resize(size, 0.2f);  // dummy input
        cudaMemcpy(dptr, hostInputs[i].data(), size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 输出 buffer
    for (size_t i = 0; i < outputNames.size(); ++i) {
        const std::string& name = outputNames[i];
        Dims dims = context->getTensorShape(name.c_str());
        size_t size = volume(dims);
        void* dptr;
        cudaMalloc(&dptr, size * sizeof(float));
        buffers[tensorIndexMap[name]] = dptr;
        context->setTensorAddress(name.c_str(), dptr);
        hostOutputs[i].resize(size);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    nvtxRangePush("Warmup");
    for (int i = 0; i < 10; ++i) {
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }
    nvtxRangePop();

    // Benchmark
    const int REPEAT = 100;
    std::vector<double> durations;
    durations.reserve(REPEAT);

    for (int i = 0; i < REPEAT; ++i) {
        nvtxRangePushA(("Infer #" + std::to_string(i)).c_str());
        auto t0 = Clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = Clock::now();
        nvtxRangePop();
        durations.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // 统计信息
    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    double avg = sum / durations.size();
    auto [mn, mx] = std::minmax_element(durations.begin(), durations.end());
    double sq = std::inner_product(durations.begin(), durations.end(), durations.begin(), 0.0);
    double stddev = std::sqrt(sq / durations.size() - avg * avg);

    std::cout << "\n=== Benchmark Results ===\n"
              << "Runs: " << REPEAT << "\n"
              << "Avg: " << avg << " ms\n"
              << "Min: " << *mn << " ms\n"
              << "Max: " << *mx << " ms\n"
              << "StdDev: " << stddev << " ms\n";

    // 输出推理结果
    for (size_t i = 0; i < outputNames.size(); ++i) {
        const std::string& name = outputNames[i];
        auto& buf = hostOutputs[i];
        cudaMemcpy(buf.data(), buffers[tensorIndexMap[name]], buf.size() * sizeof(float), cudaMemcpyDeviceToHost);
        Dims d = context->getTensorShape(name.c_str());
        std::cout << "Output " << name << ": shape=(";
        for (int j = 0; j < d.nbDims; ++j)
            std::cout << d.d[j] << (j + 1 < d.nbDims ? "," : "");
        std::cout << "), first values: ";
        for (int j = 0; j < std::min<size_t>(8, buf.size()); ++j)
            std::cout << buf[j] << " ";
        std::cout << "...\n";
    }

    // 清理资源
    for (void* p : buffers) if (p) cudaFree(p);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    return 0;
}



// ./trt_perf /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 1024
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_lhuc_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 2048
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_comxcom_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/comxcom/comxcom.engine 2048
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_mlcc_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/mlcc/mlcc.engine 2048
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_mmoe_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/mmoe/mmoe.engine 2048
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_ppnet_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/ppnet/ppnet.engine 2048
// nsys profile  --trace=cuda,nvtx,cudnn -o /data/app/order_large/common_infer_graph/perf/nsys_profile/trt_user_model_profile_b2048_view  ./trt_perf /data/app/order_large/common_infer_graph/model/user_model/user_model.engine 2048