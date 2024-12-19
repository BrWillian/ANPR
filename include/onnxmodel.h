//
// Created by willian on 12/8/24.
//

#ifndef ONNXMODEL_H
#define ONNXMODEL_H

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "../include/imageop.h"
#include <numeric>
#include <chrono>
#include <vector>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>

typedef struct  {
    int classId;
    float confidence;
    cv::Rect bbox;
} Detection;

class ONNXModel {
public:
    explicit ONNXModel(const std::string& model_path);

    virtual ~ONNXModel();

    float* infer(const std::vector<float>& input_data);

    std::vector<Detection> postProcess(float* input_data) const;

    virtual void WarmUpSession();

protected:
    Ort::Session session_;
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    std::vector<std::string> classes {};

    float confThreshold = 0.1;
    float iouThreshold = 0.5;

    void initializeIO();

    Ort::Value createTensor(const std::vector<float>& data, const std::vector<int64_t>& dimensions) const;
};



#endif //ONNXMODEL_H
