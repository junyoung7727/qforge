"""
Decision Transformer 모델 사이즈 표준화 설정
S, M, L 사이즈로 표준화된 모델 구성
"""

from typing import Dict, Any


# 표준화된 모델 사이즈 설정
MODEL_CONFIGS = {
    'S': {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'max_seq_length': 64,
        'dropout': 0.1
    },
    'M': {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 128,
        'dropout': 0.1
    },
    'L': {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 8,
        'd_ff': 3072,
        'max_seq_length': 256,
        'dropout': 0.1
    }
}


def get_model_config(size: str) -> Dict[str, Any]:
    """모델 사이즈에 따른 설정 반환"""
    size = size.upper()
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size: {size}. Available sizes: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[size].copy()


def get_available_sizes() -> list:
    """사용 가능한 모델 사이즈 목록 반환"""
    return list(MODEL_CONFIGS.keys())


def print_model_info(size: str):
    """모델 사이즈 정보 출력"""
    config = get_model_config(size)
    print(f"📊 Decision Transformer Model Size: {size}")
    print(f"   - Model Dimension: {config['d_model']}")
    print(f"   - Attention Heads: {config['n_heads']}")
    print(f"   - Layers: {config['n_layers']}")
    print(f"   - Feed Forward: {config['d_ff']}")
    print(f"   - Max Sequence: {config['max_seq_length']}")
    print(f"   - Dropout: {config['dropout']}")
    
    # 대략적인 파라미터 수 계산
    approx_params = (
        config['d_model'] * config['d_model'] * 4 * config['n_layers'] +  # Attention
        config['d_model'] * config['d_ff'] * 2 * config['n_layers']       # FFN
    )
    print(f"   - Approx Parameters: {approx_params:,}")
