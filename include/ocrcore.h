//
// Created by willian on 12/19/24.
//

#ifndef OCRCORE_H
#define OCRCORE_H

#include "platereader.h"
#include "platefinder.h"
#include <regex>


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


class OcrCore {
private:
    PlateFinder *plateFinder{};
    PlateReader *plateReader{};

    const cv::Size inputPlateFinderModel = cv::Size(320, 320);
    const cv::Size inputPlateReaderModel = cv::Size(320, 320);

    std::vector<Detection> performPlateFinder(const cv::Mat &frame) const;
    std::vector<Detection> performPlateReader(const cv::Mat &frame) const;

    std::string assembleStrPlate(const std::vector<Detection> &chars) const;

    static std::optional<bool> validateLicensePlate(const std::string &plateStr);
    static std::vector<Detection> buildPlate(const std::vector<Detection> &chars, const cv::Mat &image_roi);


public:
    OcrCore();
    OcrCore(const std::string &plateModelPath, const std::string &plateReaderPath);
    ~OcrCore();

    //td::vector<std::string> getOcr(const cv::Mat &frame) const;
    std::vector<OcrResult> getOcr(const cv::Mat &frame) const;

    // OCR UTILS
    static bool compareByLength(const Detection& a, const Detection& b);
    static bool compareByConfidence(const Detection& a, const Detection& b);
    static bool compareByHeight(const Detection& a, const Detection& b);
    static void checkBbox(const cv::Mat& frame, cv::Rect& bbox);
};


#endif //OCRCORE_H
