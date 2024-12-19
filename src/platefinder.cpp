//
// Created by willian on 12/11/24.
//

#include "../include/platefinder.h"

PlateFinder::PlateFinder(const std::string& model_path)
    : ONNXModel(model_path) {
    classes = {"plate"};
}

PlateFinder::~PlateFinder() = default;
