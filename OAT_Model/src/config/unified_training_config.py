"""
Singleton Config System - 3개
통합 학습 설정 관리
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch


class ConfigSingleton:
    """싱글톤 베이스 클래스"""
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]


@dataclass
class PropertyConfig(ConfigSingleton):
    """Property Prediction Transformer 전용 설정 (싱글톤) - GPU 최적화"""
    # 디바이스 설정 (CPU 테스트용)
    device: str = "cpu"  # CPU 테스트용으로 변경
    
    # 모델 아키텍처 - 체크포인트와 호환되도록 설정
    d_model: int = 256  # 통합된 기본값
    n_heads: int = 8
    n_layers: int = 6  # 통합된 기본값
    d_ff: int = 1024
    dropout: float = 0.1  # 통합된 기본값
    attention_mode: str = "advanced"
    use_rotary_pe: bool = True
    cross_attention_heads: int = 4  # SOTA 설정
    
    # Property 특화
    property_dim: int = 3  # entanglement, fidelity, expressibility (통합)
    max_qubits: int = 10  # 통합된 기본값
    max_gates: int = 100  # 통합된 기본값
    
    # 학습 설정 (통합된 기본값)
    learning_rate: float = 1e-4  # 통합된 기본값
    min_learning_rate: float = 1e-6  # 최소 학습률
    train_batch_size: int = 32  # 통합된 기본값
    val_batch_size: int = 64  # 통합된 기본값
    grad_accum_steps: int = 4  # 그래디언트 축적 스텝 (effective batch = 16)
    weight_decay: float = 1e-4  # 더 약한 정규화
    num_epochs: int = 100
    scheduler_type: str = "cosine_with_restarts"  # 주기적 재시작
    warmup_steps: int = 500  # 워밍업 추가
    patience: int = 10  # 조기 종료 방지
    
    # 가중치 - 균형 조정 (적응형)
    entanglement_weight: float = 1.0
    fidelity_weight: float = 1.0
    expressibility_weight: float = 2.0  # 더 어려운 태스크에 높은 가중치
    
    # SOTA 설정
    consistency_loss_weight: float = 0.1
    numerical_stability: bool = True
    
    # 적응형 학습 설정
    use_adaptive_weights: bool = True  # 손실 기반 가중치 조정
    gradient_clipping: float = 1.0  # 그래디언트 클리핑
    label_smoothing: float = 0.1  # 라벨 스무딩
    
    def get_device(self) -> str:
        """실제 디바이스 반환 (GPU 우선)"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class DecisionConfig(ConfigSingleton):
    """Decision Transformer 전용 설정 (싱글톤) - GPU 최적화"""
    # 디바이스 설정 (GPU 우선)
    device: str = "cuda"  # CPU 테스트용으로 변경

    entanglement_weight: float = 10.0
    fidelity_weight: float = 0.1
    expressibility_weight: float = 0.1
    
    
    # 모델 아키텍처
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    attention_mode: str = "advanced"
    
    # Decision Transformer 특화
    max_seq_len: int = 500
    action_dim: int = 20
    state_dim: int = 50
    max_qubits: int = 50  # GPU 메모리 고려
    n_gate_types: int = 20  # 게이트 타입 수
    
    # 학습 설정 (GPU 최적화)
    learning_rate: float = 5e-4
    train_batch_size: int = 256  # 메모리 오류 방지를 위해 감소
    val_batch_size: int = 256
    grad_accum_steps: int = 4  # 그래디언트 축적 스텝 (effective batch = 16)
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    use_rotary_pe: bool = True
    
    # GPU 최적화 설정
    use_amp: bool = True  # Automatic Mixed Precision
    pin_memory: bool = True  # DataLoader 최적화
    num_workers: int = 4  # 멀티프로세싱
    
    # Logging 설정
    use_wandb: bool = True
    wandb_project: str = "quantum-decision-transformer"
    log_interval: int = 100
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    
    # 검증 및 저장
    log_every_n_steps: int = 100
    val_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    save_dir: str = "checkpoints"
    memory_cleanup_interval = 100
    
    def get_device(self) -> str:
        """실제 디바이스 반환 (GPU 우선)"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class DefaultConfig(ConfigSingleton):
    """기본 공통 설정 (싱글톤) - GPU 최적화"""
    # 디바이스 (CPU 테스트용)
    device: str = "cpu"  # CPU 테스트용으로 변경
    
    # 데이터
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # CPU 테스트 설정 (GPU 최적화 구조 유지)
    use_amp: bool = False  # CPU에서는 AMP 비활성화
    pin_memory: bool = False  # CPU에서는 pin_memory 비활성화
    num_workers: int = 2  # CPU용 감소
    prefetch_factor: int = 2  # 유지
    
    # 배치 크기 (CPU 메모리에 맞게 조정)
    train_batch_size: int = 16  # CPU용 감소
    val_batch_size: int = 16   # CPU용 감소
    
    # 캐싱 (GPU 메모리 고려)
    enable_cache: bool = True
    cache_dir: str = "cache"
    max_cache_size_gb: float = 4.0  # GPU 메모리 제한
    
    def get_device(self) -> str:
        """실제 디바이스 반환 (GPU 우선)"""
        if self.device == "auto":
            if torch.cuda.is_available():
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
                return "cuda"
            else:
                print("⚠️ CUDA not available, falling back to CPU")
                return "cpu"
        return self.device


# 싱글톤 인스턴스 접근 함수들
def get_property_config() -> PropertyConfig:
    """Property 설정 싱글톤 반환"""
    return PropertyConfig()

def get_decision_config() -> DecisionConfig:
    """Decision 설정 싱글톤 반환"""
    return DecisionConfig()

def get_default_config() -> DefaultConfig:
    """기본 설정 싱글톤 반환"""
    return DefaultConfig()


# 기존 호환성을 위한 래퍼들
def create_property_prediction_config(size: str = "medium", attention_mode: str = "advanced") -> PropertyConfig:
    """사용자 인자에 맞춰 Property 설정 생성"""
    from config.experiment_configs import MODEL_SIZES
    
    config = get_property_config()
    
    # 사용자가 지정한 크기에 맞춰 설정 업데이트
    if size in MODEL_SIZES:
        size_config = MODEL_SIZES[size]
        config.d_model = size_config["d_model"]
        config.n_heads = size_config["n_heads"] 
        config.n_layers = size_config["n_layers"]
        config.d_ff = size_config["d_ff"]
        config.dropout = size_config["dropout"]
        config.train_batch_size = size_config["batch_size"]
        config.val_batch_size = size_config["batch_size"]
    
    # 어텐션 모드 설정
    config.attention_mode = attention_mode
    
    return config


# 레거시 클래스들 (기존 코드 호환성)
ModelArchitectureConfig = DecisionConfig
TrainingConfig = DecisionConfig
PropertyPredictionConfig = PropertyConfig


@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset paths
    data_path: str = "dummy_experiment_results.json"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data processing
    max_circuit_length: int = 1000
    normalize_targets: bool = True
    augment_data: bool = False
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "cache"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: list = field(default_factory=list)
    
    # Output directories
    output_dir: str = "experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class UnifiedTrainingConfig:
    """통합 학습 설정"""
    
    # 모델 설정
    model: DecisionConfig = field(default_factory=lambda: get_decision_config())
    
    # 학습 설정
    training: DecisionConfig = field(default_factory=lambda: get_decision_config())
    
    # 데이터 설정
    data: DataConfig = field(default_factory=DataConfig)
    
    # 실험 설정
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # RTG 설정
    enable_rtg: bool = False
    property_model_size: str = "small"
    property_attention_mode: str = "advanced"
    
    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UnifiedTrainingConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        config = cls()
        
        if 'model' in data:
            config.model = ModelArchitectureConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'decision_transformer' in data:
            config.decision_transformer = DecisionTransformerConfig(**data['decision_transformer'])
        if 'property_predictor' in data:
            config.property_predictor = PropertyPredictorConfig(**data['property_predictor'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])
        
        return config
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def get_model_config_for_decision_transformer(self) -> Dict[str, Any]:
        """Get model configuration for Decision Transformer"""
        return {
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'dropout': self.model.dropout,
            'max_qubits': self.model.max_qubits,
            'n_gate_types': self.model.n_gate_types,
            'attention_mode': self.model.attention_mode,
            'device': self.model.get_device()
        }
    
    def get_model_config_for_property_predictor(self) -> Dict[str, Any]:
        """Get model configuration for Property Predictor"""
        return {
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'dropout': self.model.dropout,
            'max_qubits': self.model.max_qubits,
            'max_gates': self.model.max_gates,
            'device': self.model.get_device()
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.experiment.output_dir,
            self.experiment.checkpoint_dir,
            self.experiment.log_dir,
            self.data.cache_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def set_seed(self):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        
        random.seed(self.experiment.seed)
        np.random.seed(self.experiment.seed)
        torch.manual_seed(self.experiment.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.seed)
            torch.cuda.manual_seed_all(self.experiment.seed)
        
        if self.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Predefined experiment configurations
def get_small_experiment_config() -> UnifiedTrainingConfig:
    """Small experiment for quick testing"""
    config = UnifiedTrainingConfig()
    
    # Small model
    config.model.d_model = 256
    config.model.n_layers = 4
    config.model.n_heads = 4
    config.model.d_ff = 1024
    
    # Fast training
    config.training.num_epochs = 50
    config.training.train_batch_size = 64
    config.training.val_every_n_steps = 100
    config.training.save_every_n_steps = 200
    
    config.experiment.experiment_name = "small_test"
    config.experiment.description = "Small model for quick testing"
    
    return config


def get_medium_experiment_config() -> UnifiedTrainingConfig:
    """Medium experiment for development"""
    config = UnifiedTrainingConfig()
    
    # Medium model (default values are already medium)
    config.experiment.experiment_name = "medium"
    config.experiment.description = "Medium model for development"
    
    return config


def get_large_experiment_config() -> UnifiedTrainingConfig:
    """Large experiment for production"""
    config = UnifiedTrainingConfig()
    
    # Large model
    config.model.d_model = 768
    config.model.n_layers = 12
    config.model.n_heads = 12
    config.model.d_ff = 3072
    config.model.max_qubits = 16
    
    # Intensive training
    config.training.num_epochs = 200
    config.training.train_batch_size = 64
    config.training.learning_rate = 5e-5
    config.training.warmup_steps = 2000
    
    config.experiment.experiment_name = "large_production"
    config.experiment.description = "Large model for production use"
    
    return config


def get_config_by_name(name: str) -> UnifiedTrainingConfig:
    """Get predefined configuration by name"""
    configs = {
        'small': get_small_experiment_config,
        'medium': get_medium_experiment_config,
        'large': get_large_experiment_config
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config name: {name}. Available: {list(configs.keys())}")
    
    return configs[name]()


# Configuration manager class
class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: UnifiedTrainingConfig, name: str):
        """Save configuration with a name"""
        config_path = self.config_dir / f"{name}.json"
        config.save(config_path)
        print(f"Configuration saved to {config_path}")
    
    def load_config(self, name: str) -> UnifiedTrainingConfig:
        """Load configuration by name"""
        config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return UnifiedTrainingConfig.load(config_path)
    
    def list_configs(self) -> list:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.json")]
    
    def create_experiment_config(self, 
                               base_config: str = "medium",
                               experiment_name: str = None,
                               overrides: Dict[str, Any] = None) -> UnifiedTrainingConfig:
        """Create experiment configuration with overrides"""
        # Get base configuration
        config = get_config_by_name(base_config)
        
        # Set experiment name
        if experiment_name:
            config.experiment.experiment_name = experiment_name
        
        # Apply overrides
        if overrides:
            config.update_from_dict(overrides)
        
        # Setup experiment
        config.setup_directories()
        config.set_seed()
        
        return config


if __name__ == "__main__":
    # Example usage
    print("🔧 Unified Training Configuration System")
    
    # Create config manager
    manager = ConfigManager()
    
    # Create and save different experiment configs
    small_config = get_small_experiment_config()
    manager.save_config(small_config, "small_test")
    
    medium_config = get_medium_experiment_config()
    manager.save_config(medium_config, "medium_dev")
    
    large_config = get_large_experiment_config()
    manager.save_config(large_config, "large_production")
    
    print(f"Available configurations: {manager.list_configs()}")
    
    # Example of creating custom experiment
    custom_config = manager.create_experiment_config(
        base_config="medium",
        experiment_name="custom_experiment",
        overrides={
            "model": {"d_model": 384, "n_layers": 8},
            "training": {"learning_rate": 2e-4, "num_epochs": 50}
        }
    )
    
    print(f"Custom experiment created: {custom_config.experiment.experiment_name}")
    print(f"Model d_model: {custom_config.model.d_model}")
    print(f"Training epochs: {custom_config.training.num_epochs}")
