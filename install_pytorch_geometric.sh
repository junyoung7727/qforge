#!/bin/bash

# PyTorch Geometric 올바른 설치 스크립트
echo "=== PyTorch Geometric 설치 시작 ==="

# 시스템 정보 확인
echo "Python 버전 확인:"
python3 --version

echo "CUDA 확인:"
nvidia-smi || echo "CUDA 없음 - CPU 버전으로 설치"

# 기존 패키지 정리
echo "1. 기존 torch 관련 패키지 제거..."
pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pyg_lib 2>/dev/null || true

# PyTorch 설치 (사용 가능한 최신 버전)
echo "2. PyTorch 설치 중..."

# CUDA 사용 가능한 경우
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 감지됨 - CUDA 버전 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "CUDA 없음 - CPU 버전 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# PyTorch 설치 확인
echo "3. PyTorch 설치 확인:"
python3 -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
"

# PyTorch Geometric 의존성 먼저 설치
echo "4. PyTorch Geometric 의존성 설치..."
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__.split('+')[0])")+$(python3 -c "import torch; print('cu121' if torch.cuda.is_available() else 'cpu')").html

# PyTorch Geometric 설치
echo "5. PyTorch Geometric 설치..."
pip install torch-geometric

# 나머지 requirements.txt 패키지 설치
echo "6. 나머지 패키지 설치..."
cd ~/OAT_Model
pip install transformers numpy scipy scikit-learn matplotlib wandb python-dotenv

echo "7. 설치 완료! 테스트 중..."
python3 -c "
import torch
import torch_geometric
print('✅ 모든 패키지 설치 성공!')
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
"