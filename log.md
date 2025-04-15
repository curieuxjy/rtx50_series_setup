**PyTorch 기반의 최소 환경(minimal setup)**만 구성하고 싶으시다면, 아래와 같이 크게 세 가지 범주를 정리할 수 있습니다:

### ✅ **[목표]**
- PyTorch 기반 실험 가능 (CUDA 12.8 포함)
- 데이터 전처리 및 시각화 정도 가능
- 불필요한 모델 추론, 서버, UI, 확장 기능 제거

---

### 🔍 **최소 설치 버전 구성 기준**

| 패키지명              | 필요 여부 | 설명 |
|----------------------|-----------|------|
| `torch`              | ✅ 필수   | PyTorch 자체 |
| `numpy`              | ✅ 필수   | 텐서 연산 등 기본 수학 |
| `scipy`              | ⚠️ 선택   | 고급 수학 연산 |
| `pillow`             | ⚠️ 선택   | 이미지 로딩/변환 |
| `tqdm`               | ✅ 추천   | 진행률 표시 |
| `pyyaml`             | ✅ 추천   | 설정 파일 처리 |
| `einops`             | ✅ 추천   | 텐서 재배열 (모델 실험 시 유용) |

---

### 🪄 **최소 requirements_versions.txt 예시**

```txt
numpy==1.26.4
torch==2.8.0.dev20250324+cu128
pillow==10.4.0
tqdm==4.66.4
pyyaml==6.0.1
einops==0.8.0
```

---

### 🧹 **제거 가능한 패키지 목록과 이유**

| 패키지명 | 제거 이유 |
|----------|-----------|
| `transformers`, `tokenizers` | HuggingFace 관련, 모델이 필요할 경우만 설치 |
| `gradio`, `httpx` | UI 및 서버 API용 |
| `opencv-contrib-python-headless` | 고급 영상 처리용, 테스트에선 불필요 |
| `pytorch_lightning`, `omegaconf`, `accelerate` | 고급 학습 프레임워크/설정 관리용 |
| `psutil` | 시스템 정보 수집용 |
| `pygit2` | git 연동 자동화 기능 |
| `safetensors` | 모델 저장 포맷. 모델 추론 없으면 제거 가능 |
| `rembg`, `groundingdino-py`, `segment_anything`, `onnxruntime`, `timm` | 특화된 모델 추론 관련 기능들로, 미사용 시 불필요 |

---

### 🧪 만약 학습 및 실험도 원한다면?

- `pytorch_lightning`, `omegaconf`, `scipy`, `accelerate` 정도를 추가할 수 있습니다.
- 또는 직접 `.py` 파일에서 로직 구성이 가능하다면 굳이 추가할 필요는 없습니다.

---

### 💬 요약

> 최종 최소 설치용 requirements 파일은 다음과 같이 줄일 수 있습니다:

```txt
torch==2.8.0.dev20250324+cu128
numpy==1.26.4
pillow==10.4.0
tqdm==4.66.4
pyyaml==6.0.1
einops==0.8.0
```

필요하면 나중에 추가로 다시 구성 가능하니, 위 버전으로 먼저 세팅해보시면 됩니다. 원하시면 이후의 테스트용 `run.py` 예제 코드도 만들어드릴 수 있어요.