//
// Created by willian on 12/18/24.
//

#ifndef PLATEREADER_H
#define PLATEREADER_H

#include "onnxmodel.h"

class PlateReader final : public ONNXModel {
public:
    explicit PlateReader(const std::string &model_path);
    explicit PlateReader(const unsigned char model_weights[], const unsigned int model_weights_size);
    ~PlateReader() override;

    void setClasses(const std::vector<std::string> &classes);
    std::vector<std::string> getClasses();
};



#endif //PLATEREADER_H
