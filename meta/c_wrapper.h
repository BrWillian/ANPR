//
// Created by willian on 12/23/24.
//

#ifndef C_WRAPPER_H
#define C_WRAPPER_H


#ifdef __cplusplus
#include <string>
#include "../include/ocr_core.h"
#include <opencv2/opencv.hpp>
#endif

#if defined(__GNUC__)
//  GCC
#define ANPR_API __attribute__((visibility("default")))
#define IMPORT
#define CDECL __attribute__((cdecl))
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#define CDECL
#pragma warning Unknown dynamic link import/export semantics.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
    struct ANPRDetect {
        OcrCore* ocrCore;
        std::vector<OcrResult>* plates;
        cv::Mat* image;
        std::stringstream* ss;
    };
#else
    struct ANPRDetect {
        void* ocrCore;
        void* plates;
        void* image;
        void* ss;
    };
#endif

    typedef struct ANPRDetect anpr_t;

    ANPR_API anpr_t* C_ANPRCREATE();

    ANPR_API void C_ANPRDELETE(anpr_t* anpr);

    ANPR_API const char* C_ANPRINFERENCE(anpr_t* anpr, unsigned char* imgData, int imgSize);

    ANPR_API const char* C_ANPRVERSION();

#ifdef __cplusplus

    ANPR_API std::string CPP_ANPRINFER(anpr_t* anpr, cv::Mat& img);

    ANPR_API std::string CPP_ANPRVERSION();

#endif

#ifdef __cplusplus
}
#endif

#endif //C_WRAPPER_H
