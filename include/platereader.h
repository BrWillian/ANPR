//
// Created by willian on 12/18/24.
//

#ifndef OCR_H
#define OCR_H

#include "onnxmodel.h"

class Ocr final : public ONNXModel {
public:
    explicit Ocr(const std::string &model_path);
    ~Ocr() override;
};



#endif //OCR_H
