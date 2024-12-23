//
// Created by willian on 12/11/24.
//

#include "../include/platefinder.h"

PlateFinder::PlateFinder(const std::string& model_path)
    : ONNXModel(model_path) {
    this->classes = {"plate"};
}

PlateFinder::~PlateFinder() = default;

void PlateFinder::setClasses(const std::vector<std::string> &classes) {
   std::ranges::copy(classes, this->classes.begin());
}

std::vector<std::string> PlateFinder::getClasses() {
    return this->classes;
}
