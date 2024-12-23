#include <iostream>
#include "../include/platefinder.h"
#include "../include/platereader.h"
#include "../include/imageop.h"
#include "../include/ocrcore.h"


int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <image.jpg>" << std::endl;
        return 1;
    }
    OcrCore *ocr = new OcrCore();

    cv::Mat inputImage = cv::imread(argv[1]);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    auto results = ocr->getOcr(inputImage);

    for (auto &res : results) {
        std::cout<<"Plate: "<<res.plateString<<std::endl;
        std::cout<<"Conf: "<<res.confidence<<std::endl;
        std::cout<<"Bbox: "<<res.rect<<std::endl;
        std::cout<<"Mercosul: "<<res.isMercosul<<std::endl;
        std::cout<<"---------Chars Debug---------"<<std::endl;
        for (auto &p : res.chars) {
            std::cout<<p.letter<<" " <<p.confidence<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"---------Chars Debug---------"<<std::endl;
    }

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

    std::cout<<"Processed time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<"ms"<<std::endl;

    return 0;
}
