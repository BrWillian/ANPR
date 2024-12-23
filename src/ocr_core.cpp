//
// Created by willian on 12/19/24.
//

#include "../include/ocr_core.h"
//#include "../generated/weights.h"

OcrCore::OcrCore() {
    plateFinder = new PlateFinder(plate_onnx, plate_onnx_len);
    plateReader = new PlateReader(ocr_onnx, ocr_onnx_len);
}
OcrCore::OcrCore(const std::string &plateModelPath, const std::string &plateReaderPath) {
    plateFinder = new PlateFinder(plateModelPath);
    plateReader = new PlateReader(plateReaderPath);
}

OcrCore::~OcrCore() = default;

std::vector<OcrResult> OcrCore::getOcr(const cv::Mat &frame) const {
    std::vector<OcrResult> result{};
    std::vector<CharResult> charsResults{};

    auto plates = this->performPlateFinder(frame);

    for (auto &plate : plates) {
        plate.bbox.x = std::max(0, plate.bbox.x - plate.bbox.width / 5);
        plate.bbox.y = std::max(0, plate.bbox.y - static_cast<int>(plate.bbox.height / 1.5));
        plate.bbox.width += plate.bbox.width * 2 / 5;
        plate.bbox.height += plate.bbox.height * (2 / 1.5);
        plate.bbox.height = std::min(plate.bbox.height, frame.rows - plate.bbox.y);
        plate.bbox.width = std::min(plate.bbox.width, frame.cols - plate.bbox.x);

        OcrCore::checkBbox(frame, plate.bbox);
        cv::Mat image_roi = frame(plate.bbox).clone();

        std::vector<Detection> chars = this->performPlateReader(image_roi);

        auto plateChars = this->buildPlate(chars, image_roi);

        auto plateStr = this->assembleStrPlate(plateChars);

        auto validatedPlate = this->validateLicensePlate(plateStr);

        if (validatedPlate.has_value()) {
            charsResults.clear();
            for (size_t i = 0; i < plateChars.size(); i++) {
                charsResults.push_back({
                    plateStr[i],
                    plateChars[i].confidence,
                    plateChars[i].bbox,
                });
            }

            result.push_back({
                plateStr,
                validatedPlate.value(),
                plate.confidence,
                plate.bbox,
                charsResults
            });
        }
    }

    return result;
}

std::vector<Detection> OcrCore::performPlateFinder(const cv::Mat &frame) const {
    const cv::Mat processedImage = ImageOperator::preprocess(frame, inputPlateFinderModel);
    const std::vector<float> blobImage = ImageOperator::toBlob(processedImage);
    float* output_tensor = plateFinder->infer(blobImage);
    return plateFinder->postProcess(output_tensor);
}

std::vector<Detection> OcrCore::performPlateReader(const cv::Mat &frame) const {
    const cv::Mat processedImage = ImageOperator::preprocess(frame, inputPlateReaderModel);
    const std::vector<float> blobImage = ImageOperator::toBlob(processedImage);
    float* output_tensor = plateReader->infer(blobImage);
    return plateReader->postProcess(output_tensor);
}

std::vector<Detection> OcrCore::buildPlate(const std::vector<Detection> &chars, const cv::Mat &image_roi) {
    if (chars.size() < 7) {
        return {};
    }

    std::vector<Detection> plateTmp(chars.begin(), chars.begin() + std::min(7, static_cast<int>(chars.size())));

    if (image_roi.cols < image_roi.rows * 1.4) {
        std::ranges::sort(plateTmp, compareByHeight);

        std::vector<Detection> topRow(plateTmp.begin(), plateTmp.begin() + 3);
        std::vector<Detection> bottomRow(plateTmp.end() - 4, plateTmp.end());

        std::ranges::sort(topRow, compareByLength);
        std::ranges::sort(bottomRow, compareByLength);

        plateTmp.insert(plateTmp.end(), topRow.begin(), topRow.end());
        plateTmp.insert(plateTmp.end(), bottomRow.begin(), bottomRow.end());
    } else {
        std::ranges::sort(plateTmp, compareByLength);
    }

    return plateTmp;
}

std::string OcrCore::assembleStrPlate(const std::vector<Detection> &chars) const {
    if (chars.empty()) {
        return {};
    }

    std::string plateStr;
    for (const auto &c : chars) {
        plateStr += plateReader->getClasses()[static_cast<int>(c.classId)];
    }
    return plateStr;
}

std::optional<bool> OcrCore::validateLicensePlate(const std::string &plateStr) {
    if (plateStr.empty()) {
        return std::nullopt;
    }

    static const std::regex oldFormat("^[A-Z]{3}\\d{4}$");
    static const std::regex newFormat("^[A-Z]{3}\\d[A-Z]\\d{2}$");

    if (std::regex_match(plateStr, newFormat)) {
        return true; // Mercosul
    }

    if (std::regex_match(plateStr, oldFormat)) {
        return false; // Normal
    }

    return std::nullopt; // Inválido
}

inline bool OcrCore::compareByLength(const Detection &a, const Detection &b) {
    return a.bbox.x < b.bbox.x;
}

inline bool OcrCore::compareByConfidence(const Detection &a, const Detection &b) {
    return a.confidence > b.confidence;
}

inline bool OcrCore::compareByHeight(const Detection &a, const Detection &b) {
    return a.bbox.y < b.bbox.y;
}

inline void OcrCore::checkBbox(const cv::Mat &frame, cv::Rect &bbox) {
    const int x = std::max(bbox.x, 0);
    const int y = std::max(bbox.y, 0);
    const int width = std::min(bbox.width, frame.cols - x);
    const int height = std::min(bbox.height, frame.rows - y);
    bbox = (width <= 0 || height <= 0) ? cv::Rect(1,1,1,1) : cv::Rect(x, y, width, height);
}