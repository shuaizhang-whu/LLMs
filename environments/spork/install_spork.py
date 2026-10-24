#!/usr/bin/env python3
"""复制这一个文件到目标设备，在 Conda 的 base 环境中运行即可安装 spork。"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


# The pinned specifications below are embedded so no companion files are needed.
SNAPSHOT = {'captured_at': '2026-10-21T19:49:46.624407+08:00',
 'python_version': '3.12.14',
 'platform': 'Linux',
 'machine': 'x86_64',
 'minimum_glibc': '2.34',
 'torch_runtime_version': '2.13.0+cu130',
 'cuda_runtime': '13.0',
 'packages': {'addict': '2.4.0',
              'agent-detector': '2.0.0',
              'aiohappyeyeballs': '2.7.1',
              'aiohttp': '3.14.3',
              'aiosignal': '1.4.0',
              'annotated-doc': '0.0.5',
              'annotated-types': '0.8.0',
              'anthropic': '1.7.0',
              'anyio': '4.15.1',
              'apache-tvm-ffi': '0.1.11',
              'astor': '0.8.1',
              'attrs': '26.1.0',
              'blake3': '1.0.9',
              'cachebox': '5.2.3',
              'cachetools': '7.2.0',
              'cbor2': '6.1.4',
              'certifi': '2026.7.22',
              'cffi': '2.1.1',
              'charset-normalizer': '3.5.1',
              'click': '8.5.0',
              'cloudpickle': '3.1.2',
              'compressed-tensors': '0.17.0',
              'cryptography': '50.0.1',
              'cuda-bindings': '13.4.2',
              'cuda-core': '1.2.0',
              'cuda-pathfinder': '1.8.2',
              'cuda-python': '13.4.1',
              'cuda-tile': '1.6.0',
              'cuda-toolkit': '13.0.3.0',
              'datasets': '5.0.1',
              'deepdiff': '9.1.0',
              'depyf': '0.20.0',
              'detect-installer': '0.2.1',
              'dill': '0.4.1',
              'dnspython': '2.8.0',
              'docstring_parser': '0.18.0',
              'einops': '0.8.2',
              'email-validator': '2.3.0',
              'fastapi': '0.136.3',
              'fastapi-cli': '0.0.32',
              'fastapi-cloud-cli': '0.26.0',
              'fastar': '0.12.0',
              'fastsafetensors': '0.4.0',
              'fastuuid': '0.14.0',
              'filelock': '4.0.1',
              'flashinfer-python': '0.6.18',
              'frozenlist': '1.8.0',
              'fsspec': '2026.6.0',
              'googleapis-common-protos': '1.75.3',
              'grpcio': '1.84.0',
              'h11': '0.16.0',
              'hf-xet': '1.6.0',
              'httpcore': '1.0.9',
              'httpcore2': '2.13.0',
              'httptools': '0.8.0',
              'httpx': '0.28.1',
              'httpx2': '2.13.0',
              'huggingface_hub': '1.32.0',
              'humming-kernels': '0.1.12',
              'idna': '3.20',
              'ijson': '3.5.1',
              'importlib_metadata': '9.0.1',
              'instanttensor': '0.2.0',
              'interegular': '0.3.3',
              'Jinja2': '3.1.6',
              'jiter': '0.17.0',
              'jmespath': '1.1.0',
              'jsonschema': '4.26.0',
              'jsonschema-specifications': '2025.9.1',
              'lark': '1.2.2',
              'litellm': '1.82.6',
              'llguidance': '1.7.6',
              'llvmlite': '0.47.0',
              'lm-format-enforcer': '0.11.3',
              'loguru': '0.7.3',
              'markdown-it-py': '4.2.0',
              'MarkupSafe': '3.0.3',
              'mcp': '2.2.0',
              'mcp-types': '2.2.0',
              'mdurl': '0.1.2',
              'mistral_common': '1.11.7',
              'ml_dtypes': '0.6.0',
              'model-hosting-container-standards': '0.1.16',
              'mpmath': '1.3.0',
              'msgspec': '0.21.1',
              'multidict': '6.9.0',
              'multiprocess': '0.70.19',
              'nccl4py': '0.5.0',
              'networkx': '3.6.1',
              'ninja': '1.13.2',
              'numba': '0.65.0',
              'numpy': '2.3.5',
              'nvidia-cublas': '13.1.1.3',
              'nvidia-cublas-cu12': '12.8.4.1',
              'nvidia-cuda-cccl': '13.3.4.3.1',
              'nvidia-cuda-crt': '13.4.92',
              'nvidia-cuda-cupti': '13.0.85',
              'nvidia-cuda-cupti-cu12': '12.8.90',
              'nvidia-cuda-nvcc': '13.4.92',
              'nvidia-cuda-nvdisasm': '13.4.92',
              'nvidia-cuda-nvrtc': '13.0.88',
              'nvidia-cuda-nvrtc-cu12': '12.8.93',
              'nvidia-cuda-runtime': '13.0.96',
              'nvidia-cuda-runtime-cu12': '12.8.90',
              'nvidia-cudnn-cu12': '9.10.2.21',
              'nvidia-cudnn-cu13': '9.20.0.48',
              'nvidia-cudnn-frontend': '1.29.0',
              'nvidia-cufft': '12.0.0.61',
              'nvidia-cufft-cu12': '11.3.3.83',
              'nvidia-cufile': '1.15.1.6',
              'nvidia-cufile-cu12': '1.13.1.3',
              'nvidia-curand': '10.4.0.35',
              'nvidia-curand-cu12': '10.3.9.90',
              'nvidia-cusolver': '12.0.4.66',
              'nvidia-cusolver-cu12': '11.7.3.90',
              'nvidia-cusparse': '12.6.3.3',
              'nvidia-cusparse-cu12': '12.5.8.93',
              'nvidia-cusparselt-cu12': '0.7.1',
              'nvidia-cusparselt-cu13': '0.8.1',
              'nvidia-cutlass-dsl': '4.6.2',
              'nvidia-cutlass-dsl-libs-base': '4.6.2',
              'nvidia-cutlass-dsl-libs-core': '4.6.2',
              'nvidia-cutlass-dsl-libs-cu12': '4.6.2',
              'nvidia-cutlass-dsl-libs-cu13': '4.6.2',
              'nvidia-ml-py': '13.595.45',
              'nvidia-nccl-cu12': '2.27.3',
              'nvidia-nccl-cu13': '2.29.7',
              'nvidia-nvjitlink': '13.4.92',
              'nvidia-nvjitlink-cu12': '12.8.93',
              'nvidia-nvshmem-cu13': '3.4.5',
              'nvidia-nvtx': '13.0.85',
              'nvidia-nvtx-cu12': '12.8.90',
              'nvidia-nvvm': '13.4.92',
              'nvitop': '1.7.1',
              'nvtx': '0.2.15',
              'openai': '3.16.2',
              'openai-harmony': '0.0.8',
              'opencv-python-headless': '5.0.0.93',
              'opentelemetry-api': '1.44.0',
              'opentelemetry-exporter-otlp': '1.44.0',
              'opentelemetry-exporter-otlp-proto-common': '1.44.0',
              'opentelemetry-exporter-otlp-proto-grpc': '1.44.0',
              'opentelemetry-exporter-otlp-proto-http': '1.44.0',
              'opentelemetry-proto': '1.44.0',
              'opentelemetry-sdk': '1.44.0',
              'opentelemetry-semantic-conventions': '0.65b0',
              'opentelemetry-semantic-conventions-ai': '0.5.1',
              'orderly-set': '5.5.0',
              'outlines_core': '0.2.14',
              'packaging': '26.3',
              'pandas': '3.0.6',
              'partial-json-parser': '0.2.1.1.post7',
              'pillow': '12.3.0',
              'pip': '26.2.1',
              'prometheus-fastapi-instrumentator': '8.1.0',
              'prometheus_client': '0.26.0',
              'propcache': '0.5.4',
              'protobuf': '6.33.6',
              'psutil': '7.2.2',
              'py-cpuinfo': '9.0.0',
              'pyarrow': '25.0.1',
              'pybase64': '1.5.0',
              'pycountry': '26.2.16',
              'pycparser': '3.0',
              'pydantic': '2.13.5',
              'pydantic-extra-types': '2.11.1',
              'pydantic-settings': '2.15.0',
              'pydantic_core': '2.46.5',
              'Pygments': '2.21.0',
              'PyJWT': '2.14.0',
              'pynvvideocodec': '2.0.4',
              'python-dateutil': '2.9.0.post0',
              'python-dotenv': '1.2.3',
              'python-json-logger': '4.2.0',
              'python-multipart': '0.0.32',
              'PyYAML': '6.0.3',
              'pyzmq': '27.2.0',
              'quack-kernels': '0.6.4',
              'referencing': '0.37.0',
              'regex': '2026.9.10',
              'requests': '2.34.2',
              'rich': '15.0.0',
              'rich-toolkit': '0.20.5',
              'rignore': '0.8.1',
              'rpds-py': '2026.6.3',
              'safetensors': '0.8.0',
              'sentencepiece': '0.2.2',
              'sentry-sdk': '2.69.2',
              'setproctitle': '1.3.7',
              'setuptools': '83.0.0',
              'shellingham': '1.5.4',
              'six': '1.17.0',
              'sniffio': '1.3.1',
              'sse-starlette': '3.4.11',
              'starlette': '1.6.0',
              'supervisor': '4.3.0',
              'sympy': '1.14.0',
              'tabulate': '0.10.0',
              'tau2': '1.0.1',
              'tenacity': '9.1.4',
              'tiktoken': '0.14.0',
              'tilelang': '0.1.12',
              'tokenizers': '0.23.2',
              'tokenspeed-mla': '0.1.8',
              'tokenspeed-triton': '3.8.10.post20260906',
              'toml': '0.10.2',
              'torch': '2.13.0',
              'torch_c_dlpack_ext': '0.1.5',
              'torchaudio': '2.11.0',
              'torchcodec': '0.16.0',
              'torchvision': '0.28.0',
              'tqdm': '4.70.1',
              'transformers': '5.17.0',
              'triton': '3.7.1',
              'truststore': '0.10.4',
              'typer': '0.27.2',
              'typing-inspection': '0.4.4',
              'typing_extensions': '4.16.0',
              'urllib3': '2.8.0',
              'uvicorn': '0.53.0',
              'uvloop': '0.22.1',
              'vllm': '0.29.0',
              'watchfiles': '1.2.0',
              'websockets': '17.1',
              'wheel': '0.47.0',
              'xgrammar': '0.2.7',
              'xxhash': '4.0.1',
              'yarl': '1.25.1',
              'z3-solver': '4.15.4.0',
              'zipp': '4.1.0'},
 'cuda_forward_compat': {'package': 'cuda-compat-13-0',
                         'version': '580.178.04-1ubuntu1',
                         'url': 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb',
                         'sha256': '14a3d14373f882297f368d6282fc7fba85e46682f34166291d61df1913a59c8f',
                         'relative_library_path': 'usr/local/cuda-13.0/compat',
                         'supported_source_driver_major': 570,
                         'supported_gpu_names': ['NVIDIA A40'],
                         'documentation': 'https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html'}}
CONDA_EXPLICIT = """# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
# created-by: conda 26.1.1
@EXPLICIT
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/_libgcc_mutex-0.1-main.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/_openmp_mutex-5.1-52_gnu.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/ca-certificates-2026.8.13-h06a4308_0.conda
https://conda.anaconda.org/nvidia/noarch/cuda-version-13.0-hc7b4dd1_4.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/kernel-headers_linux-64-4.18.0-h3108a97_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/ld_impl_linux-64-2.44-h9e0c5a2_3.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/libgcc-devel_linux-64-15.2.0-hcb64138_108.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libgomp-15.2.0-h4751f2c_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/libstdcxx-devel_linux-64-15.2.0-hcb64138_108.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/tzdata-2026c-he532380_0.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-cccl_linux-64-13.0.85-1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/cuda-crt-dev_linux-64-13.0.88-h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cuda-crt-tools-13.0.88-h06a4308_0.conda
https://conda.anaconda.org/nvidia/noarch/cuda-cudart-static_linux-64-13.0.96-hbe36340_1.conda
https://conda.anaconda.org/nvidia/noarch/cuda-cudart_linux-64-13.0.96-h73695a4_1.conda
https://conda.anaconda.org/nvidia/noarch/cuda-driver-dev_linux-64-13.0.96-hbe36340_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/cuda-nvvm-dev_linux-64-13.0.88-h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libgcc-15.2.0-h69a1729_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/libnvptxcompiler-dev_linux-64-13.0.88-h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/sysroot_linux-64-2.28-h3108a97_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/binutils_impl_linux-64-2.44-h78f17ca_3.conda
https://conda.anaconda.org/nvidia/noarch/cuda-cudart-dev_linux-64-13.0.96-hbe36340_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cuda-nvvm-impl-13.0.88-h47b2149_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cuda-nvvm-tools-13.0.88-h47b2149_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libgcc-ng-15.2.0-h166f726_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libnvptxcompiler-dev-13.0.88-h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libstdcxx-15.2.0-h39759b7_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libzlib-1.3.2-h47b2149_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/openssl-3.5.8-h1b28b03_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/pthread-stubs-0.3-h47b2149_2.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/xorg-xorgproto-2025.1-h47b2149_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/xz-5.8.2-h448239c_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/binutils_linux-64-2.44-hc03a8fd_3.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/bzip2-1.0.8-h5eee18b_6.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/cuda-nvcc-dev_linux-64-13.0.88-he2511ca_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cuda-nvcc-tools-13.0.88-h7354ed3_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libexpat-2.8.4-h7354ed3_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libffi-3.4.8-h06d3fd0_3.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libsanitizer-15.2.0-h56912e6_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libstdcxx-ng-15.2.0-hc03a8fd_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libuuid-1.41.5-h5eee18b_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/ncurses-6.6-hfaaeb4e_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/xorg-libxau-1.0.12-h9b100fa_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/xorg-libxdmcp-1.1.5-h9b100fa_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/zlib-1.3.2-h47b2149_0.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-13.0.96-ha533d76_1.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-static-13.0.96-h3b4bcfc_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/gcc_impl_linux-64-15.2.0-h36674fd_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/libxcb-1.17.0-h9b100fa_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/readline-8.3-hc2a1206_0.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-dev-13.0.96-h3b4bcfc_1.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/gcc_linux-64-15.2.0-h1ca4c6b_15.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/gxx_impl_linux-64-15.2.0-h6e002af_8.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/sqlite-3.53.4-h795bf6d_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/xorg-libx11-1.8.13-h65de747_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cuda-nvcc-impl-13.0.88-hfb20e49_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/gxx_linux-64-15.2.0-h94d4220_15.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/tk-8.6.15-h54e0aa7_0.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-nvcc_linux-64-13.0.88-he6262bf_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/python-3.12.14-hc5f7cf0_0.conda
https://conda.anaconda.org/nvidia/linux-64/cuda-nvcc-13.0.88-hb25713b_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/packaging-26.3-py312h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/setuptools-83.0.0-py312h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/wheel-0.47.0-py312h06a4308_0.conda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/pip-26.2.1-pyhc872135_0.conda
"""
PIP_REQUIREMENTS = """addict==2.4.0
agent-detector==2.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==1.7.0
anyio==4.15.1
apache-tvm-ffi==0.1.11
astor==0.8.1
attrs==26.1.0
blake3==1.0.9
cachebox==5.2.3
cachetools==7.2.0
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
cloudpickle==3.1.2
compressed-tensors==0.17.0
cryptography==50.0.1
cuda-bindings==13.4.2
cuda-core==1.2.0
cuda-pathfinder==1.8.2
cuda-python==13.4.1
cuda-tile==1.6.0
cuda-toolkit==13.0.3.0
datasets==5.0.1
deepdiff==9.1.0
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
dnspython==2.8.0
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.136.3
fastapi-cli==0.0.32
fastapi-cloud-cli==0.26.0
fastar==0.12.0
fastsafetensors==0.4.0
fastuuid==0.14.0
filelock==4.0.1
flashinfer-python==0.6.18
frozenlist==1.8.0
fsspec==2026.6.0
googleapis-common-protos==1.75.3
grpcio==1.84.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.13.0
httptools==0.8.0
httpx==0.28.1
httpx2==2.13.0
huggingface_hub==1.32.0
humming-kernels==0.1.12
idna==3.20
ijson==3.5.1
importlib_metadata==9.0.1
instanttensor==0.2.0
interegular==0.3.3
Jinja2==3.1.6
jiter==0.17.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
litellm==1.82.6
llguidance==1.7.6
llvmlite==0.47.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mcp==2.2.0
mcp-types==2.2.0
mdurl==0.1.2
mistral_common==1.11.7
ml_dtypes==0.6.0
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgspec==0.21.1
multidict==6.9.0
multiprocess==0.70.19
nccl4py==0.5.0
networkx==3.6.1
ninja==1.13.2
numba==0.65.0
numpy==2.3.5
nvidia-cublas==13.1.1.3
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cccl==13.3.4.3.1
nvidia-cuda-crt==13.4.92
nvidia-cuda-cupti==13.0.85
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvcc==13.4.92
nvidia-cuda-nvdisasm==13.4.92
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime==13.0.96
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-cu13==9.20.0.48
nvidia-cudnn-frontend==1.29.0
nvidia-cufft==12.0.0.61
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile==1.15.1.6
nvidia-cufile-cu12==1.13.1.3
nvidia-curand==10.4.0.35
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver==12.0.4.66
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse==12.6.3.3
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cusparselt-cu13==0.8.1
nvidia-cutlass-dsl==4.6.2
nvidia-cutlass-dsl-libs-base==4.6.2
nvidia-cutlass-dsl-libs-core==4.6.2
nvidia-cutlass-dsl-libs-cu12==4.6.2
nvidia-cutlass-dsl-libs-cu13==4.6.2
nvidia-ml-py==13.595.45
nvidia-nccl-cu12==2.27.3
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.4.92
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvtx-cu12==12.8.90
nvidia-nvvm==13.4.92
nvitop==1.7.1
nvtx==0.2.15
openai==3.16.2
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-grpc==1.44.0
opentelemetry-exporter-otlp-proto-http==1.44.0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
opentelemetry-semantic-conventions-ai==0.5.1
orderly-set==5.5.0
outlines_core==0.2.14
packaging==26.3
pandas==3.0.6
partial-json-parser==0.2.1.1.post7
pillow==12.3.0
pip==26.2.1
prometheus-fastapi-instrumentator==8.1.0
prometheus_client==0.26.0
propcache==0.5.4
protobuf==6.33.6
psutil==7.2.2
py-cpuinfo==9.0.0
pyarrow==25.0.1
pybase64==1.5.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.5
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pydantic_core==2.46.5
Pygments==2.21.0
PyJWT==2.14.0
pynvvideocodec==2.0.4
python-dateutil==2.9.0.post0
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
PyYAML==6.0.3
pyzmq==27.2.0
quack-kernels==0.6.4
referencing==0.37.0
regex==2026.9.10
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.5
rignore==0.8.1
rpds-py==2026.6.3
safetensors==0.8.0
sentencepiece==0.2.2
sentry-sdk==2.69.2
setproctitle==1.3.7
setuptools==83.0.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sse-starlette==3.4.11
starlette==1.6.0
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tenacity==9.1.4
tiktoken==0.14.0
tilelang==0.1.12
tokenizers==0.23.2
tokenspeed-mla==0.1.8
tokenspeed-triton==3.8.10.post20260906
toml==0.10.2
torch==2.13.0
torch_c_dlpack_ext==0.1.5
torchaudio==2.11.0
torchcodec==0.16.0
torchvision==0.28.0
tqdm==4.70.1
transformers==5.17.0
triton==3.7.1
truststore==0.10.4
typer==0.27.2
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.8.0
uvicorn==0.53.0
uvloop==0.22.1
vllm==0.29.0
watchfiles==1.2.0
websockets==17.1
wheel==0.47.0
xgrammar==0.2.7
xxhash==4.0.1
yarl==1.25.1
z3-solver==4.15.4.0
zipp==4.1.0
"""


def run(command, **kwargs):
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_tuple(value):
    return tuple(int(part) for part in value.split(".")[:2])


def writable_directory(path):
    while not path.exists():
        path = path.parent
    return path.is_dir() and os.access(str(path), os.W_OK)


def direct_environment():
    environment = os.environ.copy()
    for key in list(environment):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy") or key.startswith("PIP_"):
            environment.pop(key)
    environment.pop("CONDA_PROXY_SERVERS", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_INDEX_URL"] = "https://pypi.org/simple"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def check_routes(hosts):
    """Check visible local routes; an upstream gateway may apply other policies."""
    if not shutil.which("ip"):
        raise RuntimeError("缺少 ip 命令，无法核对公网下载出口")
    for host in sorted(set(hosts)):
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        interfaces = set()
        for address in addresses:
            result = subprocess.run(["ip", "-j", "route", "get", address], capture_output=True, text=True)
            if result.returncode:
                continue
            for route in json.loads(result.stdout):
                interface = route.get("dev", "")
                if interface.lower().startswith("zt"):
                    raise RuntimeError("{} 的出口是 {}，可能经过 ZeroTier；请先确认路由".format(host, interface))
                if interface:
                    interfaces.add(interface)
        if not interfaces:
            raise RuntimeError("没有找到 {} 的公网路由".format(host))
        print("公网下载出口：{} → {}".format(host, ", ".join(sorted(interfaces))), flush=True)


def configure_conda_direct(conda, environment, directory):
    current = json.loads(subprocess.check_output(
        [conda, "config", "--show", "proxy_servers", "--json"], text=True, env=environment
    )).get("proxy_servers") or {}
    condarc = Path(directory) / "direct-download.condarc"
    condarc.write_text(json.dumps({"proxy_servers": {key: "" for key in set(current) | {"http", "https"}}}))
    environment["CONDARC"] = str(condarc)
    effective = json.loads(subprocess.check_output(
        [conda, "config", "--show", "proxy_servers", "--json"], text=True, env=environment
    )).get("proxy_servers") or {}
    if any(effective.values()):
        raise RuntimeError("Conda 仍有强制代理配置，已停止下载；请核对网络出口")


def compatibility_needed(driver_rows):
    rows = [line.split(",", 1) for line in driver_rows.strip().splitlines()]
    if not rows or any(len(row) != 2 for row in rows):
        raise RuntimeError("无法读取 NVIDIA 驱动和显卡型号")
    majors = {int(row[0].strip().split(".", 1)[0]) for row in rows}
    if len(majors) != 1:
        raise RuntimeError("读取到了不同的驱动版本，请先检查设备状态")
    major = majors.pop()
    if major >= 580:
        return False
    compat = SNAPSHOT["cuda_forward_compat"]
    if major == compat["supported_source_driver_major"] and all(
        row[1].strip() in compat["supported_gpu_names"] for row in rows
    ):
        return True
    raise RuntimeError("此脚本支持 580+ 驱动，或 570 驱动搭配 NVIDIA A40")


def compatibility_hooks(destination):
    library = destination / ".cuda-compat/usr/local/cuda-13.0/compat"
    if not (library / "libcuda.so.1").exists():
        raise RuntimeError("兼容包内没有预期的 libcuda.so.1")
    activate = destination / "etc/conda/activate.d/zz-spork-cuda-compat.sh"
    deactivate = destination / "etc/conda/deactivate.d/zz-spork-cuda-compat.sh"
    activate.parent.mkdir(parents=True, exist_ok=True)
    deactivate.parent.mkdir(parents=True, exist_ok=True)
    activate.write_text(
        '# Enable only on the older kernel driver; skip after a driver upgrade.\n'
        '_spork_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)\n'
        '_spork_major=${_spork_driver%%.*}\n'
        'if [ -n "$_spork_major" ] && [ "$_spork_major" -lt 580 ] && [ "${_SPORK_COMPAT_ACTIVE-}" != 1 ]; then\n'
        '  export _SPORK_COMPAT_OLD_LD_SET=${LD_LIBRARY_PATH+x}\n'
        '  export _SPORK_COMPAT_OLD_LD=${LD_LIBRARY_PATH-}\n'
        '  export _SPORK_COMPAT_ACTIVE=1\n'
        '  export LD_LIBRARY_PATH="$CONDA_PREFIX/.cuda-compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        'fi\n'
        'unset _spork_driver _spork_major\n'
    )
    deactivate.write_text(
        'if [ "${_SPORK_COMPAT_ACTIVE-}" = 1 ]; then\n'
        '  if [ "${_SPORK_COMPAT_OLD_LD_SET-}" = x ]; then\n'
        '    export LD_LIBRARY_PATH="${_SPORK_COMPAT_OLD_LD-}"\n'
        '  else\n'
        '    unset LD_LIBRARY_PATH\n'
        '  fi\n'
        '  unset _SPORK_COMPAT_ACTIVE _SPORK_COMPAT_OLD_LD_SET _SPORK_COMPAT_OLD_LD\n'
        'fi\n'
    )
    return library


def verify_environment(repository, metadata_only=False):
    def normalized(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    problems = []
    if platform.python_version() != SNAPSHOT["python_version"]:
        problems.append("Python 版本不同：" + platform.python_version())
    if (platform.system(), platform.machine()) != (SNAPSHOT["platform"], SNAPSHOT["machine"]):
        problems.append("操作系统或 CPU 架构与源环境不同")
    installed = {
        normalized(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    for name, version in SNAPSHOT["packages"].items():
        actual = installed.get(normalized(name))
        if actual != version:
            problems.append("{}: 期望 {}，实际 {}".format(name, version, actual or "未安装"))
    print("Python:", platform.python_version())
    print("核对包版本数:", len(SNAPSHOT["packages"]))
    for name in ("torch", "vllm", "transformers", "tau2"):
        print(name + ":", installed.get(name, "未安装"))
    if not metadata_only and not problems:
        import torch

        available = torch.cuda.is_available()
        print("PyTorch 构建:", torch.__version__)
        print("CUDA runtime:", torch.version.cuda)
        print("CUDA 可用:", available)
        if torch.__version__ != SNAPSHOT["torch_runtime_version"]:
            problems.append("PyTorch 二进制构建与源环境不同")
        if torch.version.cuda != SNAPSHOT["cuda_runtime"]:
            problems.append("PyTorch CUDA runtime 与源环境不同")
        if not available:
            problems.append("CUDA 不可用，请检查可见显卡、驱动及兼容库")
        spec = importlib.util.find_spec("tau2")
        actual_path = Path(spec.origin).resolve() if spec and spec.origin else None
        expected = (repository / "Projects/spork/tau2-bench/src").resolve()
        print("tau2 源码:", actual_path)
        if actual_path is None or expected not in actual_path.parents:
            problems.append("tau2 没有绑定到本设备的项目源码")
    if problems:
        raise RuntimeError("检查未通过：\n" + "\n".join(problems))
    print("检查通过。未加载模型，未运行推理实验。")
    return 0


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "LLMs", help="本设备项目路径，默认 ~/LLMs")
    parser.add_argument("--name", default="spork", help="新环境名，默认 spork；不会覆盖已有环境")
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    repository = args.repo.expanduser().resolve()
    if args.verify_only:
        return verify_environment(repository, args.metadata_only)
    if args.metadata_only:
        parser.error("--metadata-only 需要与 --verify-only 一起使用")
    if args.name in ("base", "root") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise RuntimeError("请使用正常的新环境名，不可使用 base 或 root")
    tau2 = repository / "Projects/spork/tau2-bench"
    if not (tau2 / "pyproject.toml").is_file():
        raise RuntimeError("找不到本设备的 tau2 源码：{}；请用 --repo 指定 LLMs 项目路径".format(tau2))
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("此安装脚本仅用于 Linux x86_64")
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or version_tuple(libc_version) < version_tuple(SNAPSHOT["minimum_glibc"]):
        raise RuntimeError("这些二进制包要求目标 glibc 至少为 " + SNAPSHOT["minimum_glibc"])
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("缺少 nvidia-smi 命令")
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], text=True
    )
    use_compat = compatibility_needed(drivers)
    if use_compat and not shutil.which("dpkg-deb"):
        raise RuntimeError("A40 兼容方案需要 dpkg-deb 解包工具")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("请先在已经初始化 Conda 的终端运行此脚本")
    info = json.loads(subprocess.check_output([conda, "info", "--json"], text=True))
    existing = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))["envs"]
    if any(Path(path).name == args.name for path in existing):
        raise RuntimeError("环境 {} 已存在，不会覆盖。可增加 --name spork-lab 使用新名称。".format(args.name))
    directories = [Path(path).expanduser().resolve() for path in info["envs_dirs"]]
    envs_directory = next((path for path in directories if writable_directory(path)), None)
    if envs_directory is None:
        raise RuntimeError("Conda 的环境目录不可写")
    destination = envs_directory / args.name
    if destination.exists():
        raise RuntimeError("目标目录已经存在，不会覆盖：" + str(destination))
    hosts = {urlparse(line).hostname for line in CONDA_EXPLICIT.splitlines() if line.startswith("https://")}
    hosts.update(("pypi.org", "files.pythonhosted.org"))
    if use_compat:
        hosts.add(urlparse(SNAPSHOT["cuda_forward_compat"]["url"]).hostname)
    check_routes(hosts)
    with tempfile.TemporaryDirectory(prefix="install-spork-") as temporary:
        temporary = Path(temporary)
        explicit = temporary / "conda.explicit.txt"
        requirements = temporary / "requirements.txt"
        explicit.write_text(CONDA_EXPLICIT)
        requirements.write_text(PIP_REQUIREMENTS)
        environment = direct_environment()
        configure_conda_direct(conda, environment, temporary)
        deb = temporary / "cuda-compat.deb"
        if use_compat:
            compat = SNAPSHOT["cuda_forward_compat"]
            print("下载 NVIDIA 用户态兼容库：", compat["version"], flush=True)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(compat["url"], timeout=45) as response, deb.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if sha256(deb) != compat["sha256"]:
                raise RuntimeError("NVIDIA 兼容包 SHA256 校验失败")
        print("按本机固定版本清单创建环境：", destination, flush=True)
        run([conda, "create", "-y", "-p", destination, "--file", explicit], env=environment)
        python = destination / "bin/python"
        environment["PATH"] = str(destination / "bin") + os.pathsep + environment.get("PATH", "")
        environment["CONDA_PREFIX"] = str(destination)
        run([conda, "env", "config", "vars", "set", "-p", destination, "PYTHONNOUSERSITE=1"], env=environment)
        if use_compat:
            run(["dpkg-deb", "-x", deb, destination / ".cuda-compat"])
            library = compatibility_hooks(destination)
            previous = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = str(library) + (os.pathsep + previous if previous else "")
        print("安装固定版本的 Python 包，并绑定本设备的 tau2 源码。", flush=True)
        run([python, "-m", "pip", "install", "--no-deps", "-r", requirements, "-e", tau2], env=environment)
        run([python, "-m", "pip", "check"], env=environment)
        run([python, Path(__file__).resolve(), "--verify-only", "--repo", repository], env=environment)
    print("环境配置完成。以后激活：conda activate " + args.name)
    print("只检查了包版本和 CUDA 可用性，没有加载模型或启动实验。")
    return 0


def entrypoint(arguments=None):
    try:
        return main(arguments)
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print("配置未完成：" + str(error), file=sys.stderr)
        print("若已开始创建环境，未完成的新环境会保留供检查。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(entrypoint())
