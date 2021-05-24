ARG CUDA
ARG FROM_IMAGE_NAME=python:3.9-slim
FROM ${FROM_IMAGE_NAME}

WORKDIR /code
ARG CUDA

# Checking nessesaty CUDA version string params
COPY pytorch_cuda_version .
RUN pytorch_cuda_ver=$(cat pytorch_cuda_version | grep ${CUDA}) && \
    if [ -z "${pytorch_cuda_ver}" ] ; then \
        echo "\033[0;31m ERROR: Wrong or missing CUDA version provided! Provide one of [ 9.2, 10.1, 10.2, 11.0 ] through --build-arg CUDA=<version> \033[0m" && \
        false; \
    fi

# Installing PyTorch. Special case for CUDA=10.2
RUN pytorch_cuda_ver=$(cat pytorch_cuda_version | grep ${CUDA} | cut -d " " -f 2) && \
    if [ -z "${pytorch_cuda_ver}" ] ; then \
        pip3 --no-cache-dir install torch==1.7.1; \
    else \
        pip3 --no-cache-dir install torch==1.7.1${pytorch_cuda_ver} -f https://download.pytorch.org/whl/torch_stable.html; \
    fi

# Packages nessesary fo T5-base
RUN pip3 --no-cache-dir install transformers wandb sentencepiece

COPY bert-base-fetch.py .
COPY mock_data.txt .

RUN python3 bert-base-fetch.py

COPY inference.py data_loader.py utils.py ./

ENTRYPOINT [ "python3", "inference.py" ]