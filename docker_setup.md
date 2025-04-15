> Reference: [Running PyTorch on RTX 5090 and 5080 GPUs](https://docs.salad.com/tutorials/pytorch-rtx5090#dockerfile)


NVIDIA의 새로운 **RTX 5090 및 RTX 5080 GPU**는 **CUDA 12.8**을 필요로 하지만, 현재 PyTorch 공식 릴리스는 이를 아직 지원하지 않습니다.  
이 가이드가 최신이 아닐 수 있으니 [여기](https://pytorch.org)에서 최신 정보를 확인해 주세요.

이 튜토리얼은 **PyTorch Nightly 빌드**를 사용하여 RTX 50-시리즈 GPU에서 PyTorch를 실행할 수 있도록 하는 우회 방법을 제공합니다.

만약 RTX 50 시리즈 GPU와 기존의 40 또는 30 시리즈 GPU 모두에서 워크로드를 실행하려 한다면, **별도의 Docker 이미지**를 유지해야 합니다.  
작성 시점 기준으로, **기존 GPU는 CUDA 12.8을 지원하지 않기 때문에**, RTX 50 시리즈용으로 빌드한 이미지는 구형 GPU에서는 작동하지 않습니다.


# 🐳 Dockerfile

다음 Dockerfile은 RTX 50-시리즈 GPU에서 PyTorch를 실행하기 위한 **필수 의존성들을 포함하는 컨테이너 환경**을 설정합니다.

- 기본 베이스 이미지는 `nvidia/cuda`의 공식 이미지이며, `CUDA 12.8.1`의 **runtime** 버전을 사용합니다.
- `devel` 버전도 존재하며, [여기서 태그 목록](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags)에서 확인할 수 있습니다.
- 이후 Python 환경을 구성하고, CUDA 12.8에 맞춘 **PyTorch Nightly 빌드**를 설치합니다.

```dockerfile
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN . /opt/venv/bin/activate

RUN pip install --upgrade pip
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

# 🛠️ Build (이미지 빌드하기)


```bash
docker build -t test/pytorch:nightly-cuda12.8-cudnn9-runtime .
```


# 🚀 사용 방법

이미지를 빌드한 후에는 기존 PyTorch 베이스 이미지처럼 `Dockerfile`에서 사용할 수 있습니다:


```bash
docker build -t test/pytorch:nightly-cuda12.8-cudnn9-runtime --push .
```


```dockerfile
# 이전 버전 예시: pytorch/pytorch:2.6.0-cuda12.6-cudnn8-runtime
FROM test/pytorch:nightly-cuda12.8-cudnn9-runtime

# 원하는 작업을 여기에 작성
```

이제 최신 **PyTorch Nightly 빌드**를 통해 **RTX 50-시리즈 GPU에서 PyTorch를 실행**할 수 있습니다!  
이 방법을 통해 최신 하드웨어에서 CUDA 12.8의 성능을 활용할 수 있습니다.

---

Docker 이미지 빌드 후 **컨테이너를 실행하고 PyTorch가 제대로 작동하는지 테스트하는 절차**를 추가한 섹션입니다. 
`docker run`, `nvidia-smi`, 그리고 테스트 스크립트 실행까지 포함되어 있습니다.


# 🧪 컨테이너 실행 및 테스트 방법

이미지를 빌드한 후, 다음과 같은 절차로 컨테이너를 실행하고 PyTorch 환경을 테스트할 수 있습니다.

## 🧱 1. 컨테이너 실행

```bash
docker run --rm -it --gpus all test/pytorch:nightly-cuda12.8-cudnn9-runtime /bin/bash
```

- `--gpus all`: Docker에서 NVIDIA GPU를 사용할 수 있도록 설정 (NVIDIA Container Toolkit 필요)
- `--rm`: 종료 시 자동 삭제
- `-it`: 대화형 터미널

> 💡 `nvidia-container-toolkit`이 설치되어 있어야 GPU가 컨테이너에서 인식됩니다. 설치 방법: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


### Error

아래와 같은 에러가 나온다면,
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

```

- cuda 설치 진행
    - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=runfile_local

```
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

```
sudo apt install nvidia-cuda-toolkit
```

---

## 🧠 2. PyTorch 테스트

컨테이너 안에서 아래 Python 코드를 입력하거나, `test_torch.py` 파일로 저장해 실행할 수 있습니다:

```bash
python
```

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Tensor test:", torch.randn(3, 3).cuda() @ torch.randn(3, 3).cuda())
```

또는 `test_torch.py`를 컨테이너 내부로 복사해서 실행:

```bash
docker cp test_torch.py <container_id>:/test_torch.py
docker exec -it <container_id> python /test_torch.py
```

---

## 🧊 3. `nvidia-smi` 확인

GPU가 정상적으로 인식되는지도 확인할 수 있습니다:

```bash
nvidia-smi
```

정상적으로 작동한다면, RTX 5090/5080이 출력되고 PyTorch에서 CUDA를 사용할 수 있어야 합니다.

---

## ✅ 예시 출력 (정상 작동 시)

```
PyTorch version: 2.8.0.dev20250324+cu128
CUDA available: True
CUDA device: NVIDIA GeForce RTX 5090
Tensor test: tensor([...], device='cuda:0')
```

---

## 📌 참고

- 만약 `torch.cuda.is_available()`이 `False`를 반환하거나 `nvidia-smi`에서 GPU가 보이지 않는다면:
  - 호스트에 NVIDIA 드라이버 또는 Container Toolkit이 제대로 설치되었는지 확인하세요.
  - Docker Desktop 사용 시 GPU 액세스 권한이 활성화되어 있는지도 확인하세요.

---

이제 여러분은 **최신 PyTorch Nightly (CUDA 12.8)** 환경을 RTX 50 시리즈 GPU에서 완전히 실행하고 검증할 수 있습니다! 🎉

> 🔗 **Reference:** [Running PyTorch on RTX 5090 and 5080 GPUs](https://docs.salad.com/tutorials/pytorch-rtx5090#dockerfile)  
> 🔗 **PyTorch Nightly CUDA 12.8 builds:** [pytorch.org](https://pytorch.org)