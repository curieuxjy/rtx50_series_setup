# Official Guide

[Set Up Your NVIDIA RTX 5090 GPU for AI Development on Linux with PyTorch](https://youtu.be/af7XjGekm4g?si=H9d-oSw84ZADM0sX)
- https://github.com/jayrodge/NVIDIA-RTX5090-AI-Dev-Setup

# Docker 

[Running PyTorch on RTX 5090 and 5080 GPUs](https://docs.salad.com/tutorials/pytorch-rtx5090#dockerfile)

[docker_setup.md](./docker_setup.md)

---


# ⚡ PyTorch CUDA 12.8 Setup for RTX 5090 (Portable Version)

## 🧠 RTX 5090 전용 PyTorch 환경 구성

이 프로젝트는 **NVIDIA RTX 5090 GPU**에서 작동하는 최신 **PyTorch Nightly (CUDA 12.8)** 환경을 간편하게 구성할 수 있도록 돕습니다.  
Python이 내장되어 있으며, 별도의 전역 Python 설치 없이도 바로 사용 가능합니다.


## ✅ 주요 특징

- **PyTorch 2.8 Nightly + CUDA 12.8** 완벽 호환
- RTX 5090에서 테스트 완료
- 내장 Python 포함 (글로벌 설치 불필요)
- 최소한의 의존성만 설치 — 빠른 설치와 낮은 용량
- 포터블 실행 — USB, 오프라인, 에어갭 환경에서도 가능

---

## 🚀 설치 방법

### 1️⃣ `install.bat` 실행

프로젝트 디렉토리에서 `install.bat` 파일을 더블 클릭하여 설치를 시작하세요.

다음 항목이 자동으로 설치됩니다:
- pip (내장 Python에 적용)
- PyTorch 2.8 Nightly (CUDA 12.8 지원)
- 최소한의 유틸리티 패키지
- 실행용 `run.bat` 생성

---

## 🔧 설치 스크립트: `install.bat`

```bat
@echo off
cd /d %~dp0
setlocal enabledelayedexpansion

:: ------------------------------------------------------------
:: STEP 1 - Install pip
echo 🔧 Step 1: Installing pip...

python_embed\python.exe python_embed\get-pip.py

if !errorlevel! neq 0 (
    echo ❌ Pip installation failed!
    pause
    exit /b 1
) else (
    echo ✅ Pip installation successful.
)

:: ------------------------------------------------------------
:: STEP 2 - Install PyTorch Nightly
echo 🧠 Step 2: Installing PyTorch Nightly with CUDA 12.8...

python_embed\python.exe -m pip install --upgrade pip

python_embed\python.exe -m pip install --pre torch==2.8.0.dev20250324+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

if !errorlevel! neq 0 (
    echo ❌ PyTorch Nightly installation failed.
    pause
    exit /b 1
) else (
    echo ✅ PyTorch Nightly installed.
)

:: ------------------------------------------------------------
:: STEP 3 - Install minimal packages
echo 📦 Step 3: Installing minimal dependencies...

python_embed\python.exe -m pip install -r minimal_requirements.txt

if !errorlevel! neq 0 (
    echo ❌ Package installation failed.
    pause
    exit /b 1
) else (
    echo ✅ All minimal packages installed.
)

:: ------------------------------------------------------------
:: STEP 4 - Create run.bat
echo 📝 Step 4: Creating run.bat...

(
echo @echo off
echo title PyTorch Environment (CUDA 12.8)
echo cd /d %%~dp0
echo python_embed\python.exe
echo pause
) > run.bat

if exist run.bat (
    echo ✅ run.bat created successfully!
) else (
    echo ❌ Failed to create run.bat
)

echo ✅ Setup complete. Run run.bat to start Python with PyTorch!
pause
```

---

## 📄 최소 의존성 목록: `minimal_requirements.txt`

```txt
torch==2.8.0.dev20250324+cu128
numpy==1.26.4
pillow==10.4.0
tqdm==4.66.4
pyyaml==6.0.1
einops==0.8.0
```

💡 추가로 필요한 패키지가 있다면 직접 `pip install`로 설치


## 🧪 테스트 환경

- ✅ NVIDIA RTX 5090
- ✅ Ubuntu 24.04
- ✅ PyTorch Nightly (CUDA 12.8 지원)
- ✅ 오프라인 실행 가능


## RUN 테스트

아래는 RTX 5090 + CUDA 12.8 환경이 제대로 작동하는지 확인할 수 있는 테스트용 PyTorch 스크립트 `test_torch.py`입니다.

### ✅ `test_torch.py` (확장 버전)

```python
import torch

def test_pytorch():
    print("🚀 [1] PyTorch 설치 및 GPU 테스트 중...\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA 사용 불가 — CPU로 실행됩니다.")

    a = torch.randn((3, 3), device=device)
    b = torch.randn((3, 3), device=device)
    c = torch.matmul(a, b)

    print("\n📐 텐서 연산 테스트 (A x B):")
    print(c)

    print("\n🧠 PyTorch 버전:", torch.__version__)
    print("📦 디바이스:", device)

def test_torchvision():
    print("\n🖼️ [2] Torchvision 테스트 중...\n")
    try:
        from torchvision import transforms, models
        dummy = torch.randn(1, 3, 224, 224)
        model = models.resnet18(weights=None)  # weights 다운 필요 없이 구조만
        out = model(dummy)
        print("✅ Torchvision 모델 연산 성공: resnet18 -> 출력 shape:", out.shape)
    except ImportError:
        print("❌ torchvision 모듈이 설치되어 있지 않습니다.")
    except Exception as e:
        print(f"❌ torchvision 테스트 중 오류 발생: {e}")

def test_torchaudio():
    print("\n🎧 [3] Torchaudio 테스트 중...\n")
    try:
        import torchaudio
        print("✅ torchaudio 설치 확인됨. 버전:", torchaudio.__version__)
        # simple transform
        waveform = torch.randn(1, 16000)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(waveform)
        print("🎵 Mel Spectrogram shape:", mel.shape)
    except ImportError:
        print("❌ torchaudio 모듈이 설치되어 있지 않습니다.")
    except Exception as e:
        print(f"❌ torchaudio 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_pytorch()
    test_torchvision()
    test_torchaudio()
```

---

### 💡 사용 방법

`run.bat`에서 아래와 같이 수정하여 자동 실행하도록 할 수도 있습니다:

```bat
@echo off
title PyTorch CUDA Test
cd /d %~dp0
python_embed\python.exe test_torch.py
pause
```

또는 기존 `run.bat`은 유지하고, `test_torch.py`를 수동으로 실행해도 됩니다:

```bash
> run.bat
>>> import test_torch
```

이 파일은 다음을 테스트합니다:

- PyTorch 설치 여부
- CUDA 12.8이 제대로 작동하는지
- 간단한 행렬 곱 연산
- 현재 사용 중인 디바이스 확인
- `torchvision`, `torchaudio` 포함 테스트

### 🔧 필요한 패키지 (이미 설치됨)

`install.bat`에서 아래 명령어를 통해 설치되도록 되어 있어야 합니다:

```bat
python_embed\python.exe -m pip install --pre torch==2.8.0.dev20250324+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```


### 🧪 이 테스트 스크립트는 확인합니다:

| 항목 | 체크 |
|------|------|
| PyTorch 설치 및 CUDA 디바이스 인식 | ✅ |
| `torchvision` 모델 실행 테스트 (ResNet18) | ✅ |
| `torchaudio` 오디오 변환 연산 (`MelSpectrogram`) | ✅ |


### run_test.bat 자동 스크립트

아래는 `test_torch.py`를 자동으로 실행하는 `.bat` 파일입니다. 

---

### ✅ `run_test.bat`

```bat
@echo off
title PyTorch + TorchVision + TorchAudio 테스트
cd /d %~dp0

echo 🧪 PyTorch 환경 테스트를 시작합니다...
echo --------------------------------------------

python_embed\python.exe test_torch.py

echo.
echo ✅ 테스트가 완료되었습니다. 창을 닫으려면 아무 키나 누르세요.
pause
```


### 💡 사용 방법

1. `test_torch.py`와 `run_test.bat` 파일을 같은 디렉토리에 둡니다.
2. `run_test.bat`을 더블 클릭하여 실행합니다.
3. 콘솔에서 PyTorch, TorchVision, Torchaudio의 정상 작동 여부가 출력됩니다.


필요하시면 `run_test.bat`이 설치 이후 자동 실행되도록 `install.bat` 마지막 단계에 아래 한 줄을 추가할 수도 있습니다:

```bat
call run_test.bat
```


## ⚖️ 라이선스

- 본 구성은 연구/실험/테스트 목적의 PyTorch 환경 구축을 위한 것입니다.
- PyTorch는 BSD 스타일 라이선스를 따릅니다.

