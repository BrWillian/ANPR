//
// Created by willian on 12/8/24.
//

#ifndef ONNXMODEL_H
#define ONNXMODEL_H

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <numeric>
#include <vector>


class ONNXModel {
public:
    explicit ONNXModel(const std::string& model_path, bool use_gpu = false);

    virtual ~ONNXModel();

    virtual std::vector<float> infer(const std::vector<float>& input_data) = 0;

    void WarmUpSession();

protected:
    Ort::Session session_;
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    bool use_gpu_;


    std::vector<int64_t> input_shape_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    size_t input_size_{};
    size_t output_size_{};

    void initializeIO();

    Ort::Value createTensor(const std::vector<float>& data, const std::vector<int64_t>& dimensions) const;
};



#endif //ONNXMODEL_H
