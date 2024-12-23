//
// Created by willian on 12/11/24.
//

#ifndef PLATEFINDER_H
#define PLATEFINDER_H

#include "onnxmodel.h"


class PlateFinder final : public ONNXModel {
public:
    explicit PlateFinder(const std::string& model_path);
    ~PlateFinder() override;

    void setClasses(const std::vector<std::string> &classes);
    std::vector<std::string> getClasses();
};

#endif //PLATEFINDER_H
