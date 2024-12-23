//
// Created by willian on 12/12/24.
//

#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>

class ImageOperator {
public:
    static cv::Mat preprocess(const cv::Mat& inputImage, const cv::Size& targetSize);
    static std::vector<float> toBlob(const cv::Mat& inputImage);

    static inline float resize_scales;
    static inline int padding;


    static inline cv::Size target_size;
    static inline cv::Size input_size;
};



#endif //IMAGE_H
