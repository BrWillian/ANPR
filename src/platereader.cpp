//
// Created by willian on 12/18/24.
//

#include "../include/platereader.h"

PlateReader::PlateReader(const std::string &model_path)
    : ONNXModel(model_path) {
    this->classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
}

PlateReader::PlateReader(const unsigned char model_weights[], const unsigned int model_weights_size)
    : ONNXModel(model_weights, model_weights_size){
    this->classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
}

PlateReader::~PlateReader() = default;

void PlateReader::setClasses(const std::vector<std::string> &classes) {
   std::ranges::copy(classes, this->classes.begin());
}

std::vector<std::string> PlateReader::getClasses() {
    return this->classes;
}
