//
// Created by willian on 12/8/24.
//

#include "../include/onnxmodel.h"

#include <iostream>

ONNXModel::ONNXModel(const std::string& model_path, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"), session_(nullptr), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), use_gpu_(use_gpu) {

    Ort::SessionOptions session_options;
    if (use_gpu_) {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    }
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(3);

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    initializeIO();
}

ONNXModel::~ONNXModel() = default;

void ONNXModel::WarmUpSession() {
}

void ONNXModel::initializeIO() {
    // Get Input Size
    size_t num_input_nodes = session_.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatedStringPtr input_node_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names_.emplace_back(input_node_name.release());
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
    }

    // Calculate input size
    input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int64_t>());

    // Get Output Size
    size_t num_output_nodes = session_.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr output_node_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names_.emplace_back(output_node_name.release());
        auto type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto output_shape = tensor_info.GetShape();
        output_size_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    }
}

Ort::Value ONNXModel::createTensor(const std::vector<float> &data, const std::vector<int64_t> &dimensions) const {
    return Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(data.data()), data.size(), dimensions.data(), dimensions.size());
}
