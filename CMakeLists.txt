cmake_minimum_required(VERSION 3.28.3)
project(ANPR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(OUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIRECTORY})

file(MAKE_DIRECTORY ${OUT_DIRECTORY})

set(ONNX_RUNTIME_LIB_STATIC /usr/local/lib/libonnxruntime.a)

if(NOT EXISTS ${ONNX_RUNTIME_LIB_STATIC})
    message(FATAL_ERROR "ONNX Runtime static library not found: ${ONNX_RUNTIME_LIB_STATIC}")
endif()

find_package(OpenCV REQUIRED)
find_package(Iconv REQUIRED)
find_package(Threads REQUIRED)

set(LIB_DEPENDENCIES ${ONNX_RUNTIME_LIB_STATIC} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

message(INFO ${LIB_DEPENDENCIES})

add_library(ANPR SHARED
        src/onnxmodel.cpp
        src/platefinder.cpp
        src/imageop.cpp
        src/platereader.cpp
        include/platereader.h
        src/ocrcore.cpp
        include/ocrcore.h
)

add_executable(debug
        main.cpp
)

target_link_libraries(ANPR PRIVATE ${LIB_DEPENDENCIES})
target_link_libraries(debug PRIVATE ANPR ${LIB_DEPENDENCIES})

set_target_properties(ANPR PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(ANPR PROPERTIES LINK_FLAGS "-Wl,-Bsymbolic-functions")

add_definitions(-O2 -pthread)
