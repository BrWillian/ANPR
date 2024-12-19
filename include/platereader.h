//
// Created by willian on 12/18/24.
//

#ifndef PLATEREADER_H
#define PLATEREADER_H

#include "onnxmodel.h"

class PlateReader final : public ONNXModel {
public:
    explicit PlateReader(const std::string &model_path);
    ~PlateReader() override;
};



#endif //PLATEREADER_H
