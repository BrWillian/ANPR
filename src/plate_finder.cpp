//
// Created by willian on 12/11/24.
//

#include "../include/plate_finder.h"

PlateFinder::PlateFinder(const std::string& model_path)
    : ONNXModel(model_path) {
    this->classes = {"plate"};
}

PlateFinder::PlateFinder(const unsigned char model_weights[], const unsigned int model_weights_size)
    : ONNXModel(model_weights, model_weights_size) {
    this->classes = {"plate"};
}

PlateFinder::~PlateFinder() = default;

void PlateFinder::setClasses(const std::vector<std::string> &classes) {
   std::ranges::copy(classes, this->classes.begin());
}

std::vector<std::string> PlateFinder::getClasses() {
    return this->classes;
}
