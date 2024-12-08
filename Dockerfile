FROM ubuntu:24.04

ARG OPENCV_VERSION=4.8.0
ARG ONNXRUNTIME_VERSION=1.20.0

RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mv opencv-${OPENCV_VERSION} opencv && \
    mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

RUN mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_SHARED_LIBS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D WITH_FFMPEG=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

WORKDIR /opt

RUN git clone --branch v${ONNXRUNTIME_VERSION} --depth 1 https://github.com/microsoft/onnxruntime.git && \
    cd /opt/onnxruntime && \
    ./build.sh --config Release  \
               --parallel \
               --skip_tests \
               --allow_running_as_root && \
    cd /opt/onnxruntime/build/Linux/Release && \
    make install && \
    ldconfig

WORKDIR /workspace

RUN rm -rf /opt/opencv* && rm -rf /opt/onnxruntime*

CMD ["/bin/bash"]
