FROM fedora:43
# make sure nvidia driver match on host and in the container
RUN dnf install -y python3-requests python3-pip gcc gcc-c++ git-core \
    && echo "[cuda-fedora43-x86_64]" >> /etc/yum.repos.d/cuda.repo \
    && echo "name=cuda-fedora43-x86_64" >> /etc/yum.repos.d/cuda.repo \
    && echo "baseurl=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64" >> /etc/yum.repos.d/cuda.repo \
    && echo "enabled=1" >> /etc/yum.repos.d/cuda.repo \
    && echo "gpgcheck=1" >> /etc/yum.repos.d/cuda.repo \
    && echo "gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/1940C73E.pub" >> /etc/yum.repos.d/cuda.repo \
    && dnf install -y cuda-compiler-13-2 cuda-toolkit-13-2 nvidia-driver-cuda nvidia-driver-cuda-libs nvidia-driver cmake \
    && dnf clean all
ENV LLAMACPP_VER="7f2cbd9a4df77a2ce10f31d69f07d4fb75eabc07"
ENV PATH=${PATH}:/usr/local/cuda-13.2/bin/

# Clone, checkout, build and move llama.cpp server to path
# for some reason, cmake doesn't pick up stuff from ENV
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout $LLAMACPP_VER && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF && \
    cmake --build build --config Release -j 4 -t llama-server && \
    mv ./build/bin/llama-server /usr/bin/llama-server
