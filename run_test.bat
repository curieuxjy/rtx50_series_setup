@echo off
title PyTorch + TorchVision + TorchAudio 테스트
cd /d %~dp0

echo 🧪 PyTorch 환경 테스트를 시작합니다...
echo --------------------------------------------

python_embed\python.exe test_torch.py

echo.
echo ✅ 테스트가 완료되었습니다. 창을 닫으려면 아무 키나 누르세요.
pause