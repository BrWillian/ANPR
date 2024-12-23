//
// Created by willian on 12/23/24.
//


#include "../meta/c_wrapper.h"
#include "../generated/version.h"

anpr_t *C_ANPRCREATE() {
    anpr_t* objwrapper;

    auto ocrCore = new OcrCore();
    auto plates = new std::vector<OcrResult>;
    auto imageContainer = new cv::Mat();
    auto strResult = new std::stringstream;

    objwrapper = (__typeof__(objwrapper)) malloc(sizeof(*objwrapper));

    objwrapper->ocrCore = ocrCore;
    objwrapper->plates = plates;
    objwrapper->image = imageContainer;
    objwrapper->ss = strResult;

    return objwrapper;
}
void CDECL C_ANPRDELETE(anpr_t* anpr){
    if(anpr == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    delete static_cast<OcrCore*>(anpr->ocrCore);
    delete static_cast<std::vector<OcrResult>*>(anpr->plates);
    delete static_cast<cv::Mat*>(anpr->image);
    free(anpr);
}
std::string Serialize(const anpr_t* anpr, const std::vector<OcrResult>& res) {
    anpr->ss->str("");
    *anpr->ss << std::boolalpha;
    *anpr->ss << "[";
    for (size_t i = 0; i < res.size(); ++i) {
        const auto&[plateString, isMercosul, confidence, rect, chars] = res[i];

        *anpr->ss << "{";
        *anpr->ss << "\"plateStr\":\"" << plateString << "\",";
        *anpr->ss << "\"isMercosul\":" << isMercosul << ",";
        *anpr->ss << "\"conf\":" << confidence << ",";
        *anpr->ss << "\"bbox\":{";
        *anpr->ss << "\"x\":" << rect.x << ",";
        *anpr->ss << "\"y\":" << rect.y << ",";
        *anpr->ss << "\"w\":" << rect.width << ",";
        *anpr->ss << "\"h\":" << rect.height;
        *anpr->ss << "},";
        *anpr->ss << "\"chars\":[";

        for (size_t j = 0; j < chars.size(); ++j) {
            const auto&[letter, confidence, rect] = chars[j];
            *anpr->ss << "{";
            *anpr->ss << "\"letter\":\"" << letter << "\",";
            *anpr->ss << "\"conf\":" << confidence << ",";
            *anpr->ss << "\"bbox\":{";
            *anpr->ss << "\"x\":" << rect.x << ",";
            *anpr->ss << "\"y\":" << rect.y << ",";
            *anpr->ss << "\"w\":" << rect.width << ",";
            *anpr->ss << "\"h\":" << rect.height;
            *anpr->ss << "}";
            *anpr->ss << "}";

            if (j != chars.size() - 1) {
                *anpr->ss << ",";
            }
        }

        *anpr->ss << "]";
        *anpr->ss << "}";

        if (i != res.size() - 1) {
            *anpr->ss << ",";
        }
    }
    *anpr->ss << "]";

    return anpr->ss->str();
}
const char* CDECL C_ANPRINFERENCE(const anpr_t* anpr, unsigned char* imgData, const int imgSize) {
    if (anpr == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    const std::vector<uchar> data(imgData, imgData + imgSize);
    *anpr->image = cv::imdecode(cv::Mat(data), -1);

    if (anpr->image->empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    try {
        *anpr->plates =  anpr->ocrCore->getOcr(*anpr->image);
    }catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    return strdup(Serialize(anpr, *anpr->plates).c_str());
}
const char* CDECL C_ANPRVERSION(){
    return ANPR_VERSION "-" GIT_BRANCH "-" GIT_COMMIT_HASH;
}
// CPP FUNCTIONS
std::string CDECL CPP_ANPRINFERENCE(const anpr_t* anpr, cv::Mat& img) {
    if (anpr == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image" << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }
    std::vector<OcrResult> plates;
    try {
        plates =  anpr->ocrCore->getOcr(img);
    }catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    return strdup(Serialize(anpr, plates).c_str());
}
std::string CDECL CPP_ANPRVERSION(){
    return ANPR_VERSION "-" GIT_BRANCH "-" GIT_COMMIT_HASH;
}