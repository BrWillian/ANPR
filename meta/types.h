//
// Created by willian on 12/22/24.
//

#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>

struct CharResult {
    char letter;
    float confidence;
    cv::Rect rect;
};

struct OcrResult {
    std::string plateString{};
    bool isMercosul{};
    float confidence{};
    cv::Rect rect;
    std::vector<CharResult> chars{};
};

struct Detection {
    int classId;
    float confidence;
    cv::Rect bbox;
};


#endif //TYPES_H
