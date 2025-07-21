#include <iostream>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <dlfcn.h>
#include <functional>
#include <NvInferPlugin.h>
#include <string>

using namespace nvinfer1;

// Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 自定义 deleter
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

// unique_ptr 类型别名
template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

template <typename T>
TRTUniquePtr<T> makeTRTUnique(T* t) {
    return TRTUniquePtr<T>(t);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <onnx_path> <engine_path> <batch_min> <batch_max>\n";
        return -1;
    }

    const std::string onnxPath = argv[1];
    const std::string enginePath = argv[2];
    int batchMin = std::stoi(argv[3]);
    int batchMax = std::stoi(argv[4]);

    // 创建 builder、network、parser…
    auto builder = makeTRTUnique(createInferBuilder(gLogger));
    uint32_t flags = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeTRTUnique(builder->createNetworkV2(flags));
    auto parser = makeTRTUnique(
        nvonnxparser::createParser(*network, gLogger));

    std::cout << "📦 TensorRT version: " << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

    // 解析 ONNX
    std::ifstream onnxFile(onnxPath, std::ios::binary);
    if (!onnxFile.good()) { std::cerr << "Cannot open ONNX file\n"; return -1; }
    std::string onnxData((std::istreambuf_iterator<char>(onnxFile)), std::istreambuf_iterator<char>());
    if (!parser->parse(onnxData.data(), onnxData.size())) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << parser->getError(i)->desc() << std::endl;
        return -1;
    }
    std::cout << "✔ ONNX parsed successfully.\n";

    // // 获取输入张量名称
    // const std::string inputName = network->getInput(0)->getName();
    // std::cout << "📥 Auto-detected input tensor: " << inputName << "\n";

    // // 获取输入的 shape
    // auto inputTensor = network->getInput(0); // 假设第一个输入是目标输入
    // if (!inputTensor || inputTensor->getName() != inputName) {
    //     std::cerr << "❌ Failed to find input tensor with name: " << inputName << "\n";
    //     return -1;
    // }
    // auto inputDims = inputTensor->getDimensions();
    // if (inputDims.nbDims < 2) {
    //     std::cerr << "❌ Input dimensions are invalid\n";
    //     return -1;
    // }

    // auto profile = builder->createOptimizationProfile();

    // // 获取输入张量的维度
    // if (inputDims.nbDims == 2) {
    //     // 如果输入张量是 2D
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims2{batchMin, inputDims.d[1]});
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims2{(batchMin + batchMax) / 2, inputDims.d[1]});
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims2{batchMax, inputDims.d[1]});
    // } else if (inputDims.nbDims == 3) {
    //     // 如果输入张量是 3D
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims3{batchMin, inputDims.d[1], inputDims.d[2]});
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims3{(batchMin + batchMax) / 2, inputDims.d[1], inputDims.d[2]});
    //     profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims3{batchMax, inputDims.d[1], inputDims.d[2]});
    // } else {
    //     std::cerr << "❌ Unsupported input tensor dimensions: " << inputDims.nbDims << "\n";
    //     return -1;
    // }

    auto profile = builder->createOptimizationProfile();

    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto inputTensor = network->getInput(i);
        std::string name = inputTensor->getName();
        auto dims = inputTensor->getDimensions();

        std::cout << "📥 Setting profile for input: " << name 
                << " with shape " << dims.nbDims << "D\n";

        if (dims.nbDims == 2) {
            profile->setDimensions(name.c_str(), OptProfileSelector::kMIN, Dims2{batchMin, dims.d[1]});
            profile->setDimensions(name.c_str(), OptProfileSelector::kOPT, Dims2{(batchMin + batchMax)/2, dims.d[1]});
            profile->setDimensions(name.c_str(), OptProfileSelector::kMAX, Dims2{batchMax, dims.d[1]});
        } else if (dims.nbDims == 3) {
            profile->setDimensions(name.c_str(), OptProfileSelector::kMIN, Dims3{batchMin, dims.d[1], dims.d[2]});
            profile->setDimensions(name.c_str(), OptProfileSelector::kOPT, Dims3{(batchMin + batchMax)/2, dims.d[1], dims.d[2]});
            profile->setDimensions(name.c_str(), OptProfileSelector::kMAX, Dims3{batchMax, dims.d[1], dims.d[2]});
        } else {
            std::cerr << "❌ Unsupported input dims: " << dims.nbDims << "\n";
            return -1;
        }
    }
    
    // BuilderConfig
    auto config = makeTRTUnique(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config->addOptimizationProfile(profile);

    // 构建引擎
    auto engineString = builder->buildSerializedNetwork(*network, *config);
    if (!engineString) { std::cerr << "❌ buildSerializedNetwork failed\n"; return -1; }

    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "❌ Failed to open engine file for writing\n";
        return -1;
    }
    engineFile.write(reinterpret_cast<const char*>(engineString->data()), engineString->size());
    std::cout << "✅ Engine saved to " << enginePath << "\n";

    return 0;
}

// ./build_engine /data/app/order_large/common_infer_graph/model/lhuc/lhuc.onnx /data/app/order_large/common_infer_graph/model/lhuc/lhuc.engine 1 2048
// ./build_engine /data/app/order_large/common_infer_graph/model/comxcom/comxcom.onnx /data/app/order_large/common_infer_graph/model/comxcom/comxcom.engine 1 2048
// ./build_engine /data/app/order_large/common_infer_graph/model/mlcc/mlcc.onnx /data/app/order_large/common_infer_graph/model/mlcc/mlcc.engine 1 2048
// ./build_engine /data/app/order_large/common_infer_graph/model/mmoe/mmoe.onnx /data/app/order_large/common_infer_graph/model/mmoe/mmoe.engine 1 2048
// ./build_engine /data/app/order_large/common_infer_graph/model/ppnet/ppnet.onnx /data/app/order_large/common_infer_graph/model/ppnet/ppnet.engine 1 2048