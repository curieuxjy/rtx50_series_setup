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