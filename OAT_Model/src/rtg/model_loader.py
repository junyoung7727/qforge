"""
Property Predictor Model Loader for RTG Calculation
가중치만으로 모델을 로드하는 효율적인 로더
"""

import torch
import torch.nn as nn
import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
    from src.config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig
    from src.config.unified_training_config import UnifiedTrainingConfig
    from src.config.experiment_configs import MODEL_SIZES
except ImportError:
    # Fallback imports for different path contexts
    try:
        from models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
        from config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig
        from config.unified_training_config import UnifiedTrainingConfig
        from config.experiment_configs import MODEL_SIZES
    except ImportError:
        # Final fallback - use relative imports from OAT_Model.src
        from OAT_Model.src.models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
        from OAT_Model.src.config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig
        from OAT_Model.src.config.unified_training_config import UnifiedTrainingConfig
        from OAT_Model.src.config.experiment_configs import MODEL_SIZES


class PropertyPredictorLoader:
    """자동으로 모델과 설정을 로드하는 간소화된 프로퍼티 프리딕터 로더"""
    
    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로 (None이면 기본 경로에서 찾기)
            device: 디바이스 ("auto", "cuda", "cpu")
        """
        if checkpoint_path is None:
            checkpoint_path = find_best_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint found in the default directory")
                
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device(device)
        self.model = None
        self.config = None
        self.checkpoint_data = None
        
    def _get_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(self) -> UnifiedPropertyPredictionTransformer:
        """체크포인트에서 모델 로드 - 항상 SOTA 아키텍처 사용"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {self.checkpoint_path}")
        
        # 체크포인트 로드
        print(f"📦 체크포인트 로딩 중: {self.checkpoint_path}")
        try:
            self.checkpoint_data = self._safe_load_checkpoint(str(self.checkpoint_path))
        except Exception as e:
            raise RuntimeError(f"체크포인트 로드 실패: {str(e)}") from e
        
        # 설정 정보 추출 및 검증
        config_data = self.checkpoint_data.get('config', {})
        if not config_data:
            print("⚠️ 체크포인트에 설정 정보가 없음, 기본 설정 사용")
        
        # 설정 생성 및 검증
        if isinstance(config_data, dict) and config_data:
            # DecisionConfig 구조 처리
            if 'model' in config_data and hasattr(config_data['model'], '__dict__'):
                # DecisionConfig 객체에서 PropertyConfig 호환 필드 추출
                decision_config = config_data['model']
                
                # 체크포인트에서 실제 아키텍처 파라미터 추출
                state_dict = self._extract_state_dict(self.checkpoint_data)
                detected_arch = self._detect_architecture_from_checkpoint(self.checkpoint_data)
                
                # FFN 크기를 state_dict에서 직접 추출
                ffn_weight_key = 'transformer_layers.0.ffn.0.weight'
                if ffn_weight_key in state_dict:
                    d_ff_actual = state_dict[ffn_weight_key].shape[0]
                else:
                    d_ff_actual = getattr(decision_config, 'd_ff', 512)
                
                # Position embedding에서 max_gates 추출
                pos_emb_key = 'circuit_embedding.position_embedding.weight'
                if pos_emb_key in state_dict:
                    max_gates_actual = state_dict[pos_emb_key].shape[0]
                else:
                    max_gates_actual = getattr(decision_config, 'max_gates', 100)
                
                config_data = {
                    'd_model': detected_arch.get('d_model', getattr(decision_config, 'd_model', 256)),
                    'n_heads': detected_arch.get('n_heads', getattr(decision_config, 'n_heads', 4)),
                    'n_layers': detected_arch.get('n_layers', getattr(decision_config, 'n_layers', 3)),
                    'd_ff': d_ff_actual,
                    'dropout': detected_arch.get('dropout', getattr(decision_config, 'dropout', 0.1)),
                    'attention_mode': getattr(decision_config, 'attention_mode', 'advanced'),
                    'use_rotary_pe': getattr(decision_config, 'use_rotary_pe', True),
                    'learning_rate': getattr(decision_config, 'learning_rate', 1e-4),
                    'train_batch_size': getattr(decision_config, 'train_batch_size', 32),
                    'val_batch_size': getattr(decision_config, 'val_batch_size', 64),
                    'weight_decay': getattr(decision_config, 'weight_decay', 1e-5),
                    'max_qubits': getattr(decision_config, 'max_qubits', 10),
                    'property_dim': 3,
                    'max_gates': max_gates_actual,
                    'cross_attention_heads': detected_arch.get('cross_attention_heads', 4),
                    'consistency_loss_weight': 0.1,
                    'numerical_stability': True,
                    'gradient_clipping': 1.0
                }
                print(f"✅ 체크포인트에서 추출된 아키텍처: d_model={config_data['d_model']}, d_ff={config_data['d_ff']}, max_gates={config_data['max_gates']}")
                print(f"✅ DecisionConfig에서 PropertyConfig 호환 설정 추출 완료")
                
                # PropertyConfig 생성 및 모델 인스턴스화
                self.config = UnifiedPropertyPredictionConfig(**config_data)
                self.model = UnifiedPropertyPredictionTransformer(self.config)
                self.model = self.model.to(self.device)
                
            elif isinstance(config_data, dict) and config_data:
                # 레거시 키 제거 (PropertyConfig와 호환되지 않는 키들)
                incompatible_keys = [
                    'model', 'data', 'training', 'evaluation', 'logging',
                    'enable_rtg', 'property_attention_mode', 'property_model_size',
                    'train_batch_size', 'val_batch_size', 'test_batch_size',
                    'learning_rate', 'weight_decay', 'warmup_steps',
                    'entanglement_weight', 'fidelity_weight', 'expressibility_weight',
                    'robust_fidelity_weight',
                    'use_wandb', 'wandb_project', 'wandb_run_name',
                    'save_interval', 'eval_interval', 'max_epochs',
                    'early_stopping_patience', 'gradient_clip_val','experiment'
                ]
                for key in incompatible_keys:
                    config_data.pop(key, None)
                
                # PropertyConfig에서 지원하지 않는 추가 키들 제거
                additional_incompatible = [
                    'optimizer', 'scheduler', 'loss_function', 'metrics',
                    'checkpoint_dir', 'log_dir', 'experiment_name',
                    'resume_from_checkpoint', 'auto_lr_find', 'precision',
                    'limit_train_batches', 'limit_val_batches', 'limit_test_batches',
                    'fast_dev_run', 'profiler', 'deterministic', 'benchmark'
                ]
                for key in additional_incompatible:
                    config_data.pop(key, None)
            
            # property_model_size와 attention_mode 처리 (이미 제거되었으므로 원본에서 가져오기)
            original_config = self.checkpoint_data.get('config', {})
            model_size = original_config.get('property_model_size', 'medium')  # 기본값을 medium으로 변경
            attention_mode = original_config.get('property_attention_mode', 'advanced')
            
            # 모델 사이즈에 따라 적절한 파라미터 설정 (experiment_configs.py에서 가져오기)
            if model_size in MODEL_SIZES:
                size_config = MODEL_SIZES[model_size]
                d_model = size_config['d_model']
                n_layers = size_config['n_layers']
                n_heads = size_config['n_heads']
                d_ff = size_config['d_ff']
                dropout = size_config['dropout']
                
            # 기본값 설정 (사용자 설정을 우선적으로 유지)
            config_data['d_model'] = config_data.get('d_model', d_model)
            config_data['n_layers'] = config_data.get('n_layers', n_layers)
            config_data['n_heads'] = config_data.get('n_heads', n_heads)
            config_data['d_ff'] = config_data.get('d_ff', d_ff)
            config_data['dropout'] = config_data.get('dropout', dropout)
            config_data['attention_mode'] = attention_mode
            
            # SOTA 설정 적용 (레거시 플래그 제거)
            config_data.update({
                'cross_attention_heads': config_data.get('cross_attention_heads', 4),
                'consistency_loss_weight': config_data.get('consistency_loss_weight', 0.1),
                'dropout': config_data.get('dropout', 0.1),
                'numerical_stability': config_data.get('numerical_stability', True),
                'gradient_clipping': config_data.get('gradient_clipping', 1.0)
            })
            
            self.config = UnifiedPropertyPredictionConfig(**config_data)
        else:
            # 기본 SOTA 설정 (medium 크기 사용)
            medium_config = MODEL_SIZES['medium']
            detected_arch = self._detect_architecture_from_checkpoint(self.checkpoint_data)
            self.config = UnifiedPropertyPredictionConfig(
                d_model=detected_arch['d_model'],
                n_layers=detected_arch['n_layers'],
                n_heads=detected_arch['n_heads'],
                d_ff=detected_arch['d_ff'],
                dropout=detected_arch['dropout'],
                cross_attention_heads=4,
                consistency_loss_weight=0.1,
                numerical_stability=True,
                gradient_clipping=1.0
            )
            print(f"🔄 모델 재생성: {detected_arch}")
            self.model = UnifiedPropertyPredictionTransformer(self.config)
        print(f"✅ 통합 아키텍처 (UnifiedPropertyPredictionTransformer)로 로드")
        print(f"🔍 모델 클래스: {self.model.__class__.__name__}")
        print(f"🔍 설정 클래스: {self.config.__class__.__name__}")
        
        # 가중치 로드 전에 실제 아키텍처 감지
        state_dict = self._extract_state_dict(self.checkpoint_data)
        detected_arch = self._detect_architecture_from_checkpoint(self.checkpoint_data)
        
        # 감지된 아키텍처와 현재 설정이 다르면 모델 재생성
        if detected_arch.get('d_model') and detected_arch['d_model'] != self.config.d_model:
            print(f"🔄 아키텍처 불일치 감지: 체크포인트 d_model={detected_arch['d_model']}, 현재 설정 d_model={self.config.d_model}")
            print("🔧 체크포인트에 맞춰 모델 재생성 중...")
            
            # 감지된 아키텍처로 설정 업데이트
            self.config.d_model = detected_arch['d_model']
            self.config.n_layers = detected_arch.get('n_layers', self.config.n_layers)
            self.config.n_heads = detected_arch.get('n_heads', self.config.n_heads)
            self.config.d_ff = detected_arch.get('d_ff', self.config.d_ff)
            
            # 모델 재생성
            self.model = UnifiedPropertyPredictionTransformer(self.config)
            self.model = self.model.to(self.device)
            print(f"✅ 모델 재생성 완료: d_model={self.config.d_model}")
        
        try:
            # 레거시 체크포인트 호환성을 위해 관대한 로딩
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # 누락된 count 버퍼들을 초기화 (backward compatibility)
            count_buffers_to_init = [
                'prediction_head.ent_count',
                'prediction_head.fid_count', 
                'prediction_head.exp_count'
            ]
            
            initialized_buffers = []
            missing_keys_list = list(missing_keys)  # Convert to list for safe removal
            for buffer_name in count_buffers_to_init:
                if buffer_name in missing_keys_list:
                    try:
                        # count 버퍼를 0으로 초기화
                        if hasattr(self.model, 'prediction_head'):
                            pred_head = self.model.prediction_head
                            buffer_attr = buffer_name.split('.')[-1]  # 'ent_count', 'fid_count', 'exp_count'
                            if hasattr(pred_head, buffer_attr):
                                getattr(pred_head, buffer_attr).fill_(0)
                                initialized_buffers.append(buffer_name)
                                missing_keys_list.remove(buffer_name)
                    except Exception as e:
                        print(f"⚠️ {buffer_name} 초기화 실패: {e}")
            
            missing_keys = set(missing_keys_list)  # Convert back to set
            
            print(f"📊 가중치 로딩 결과:")
            print(f"   - 누락된 키: {len(missing_keys)}개")
            print(f"   - 예상치 못한 키: {len(unexpected_keys)}개")
            
            if initialized_buffers:
                print(f"   - 초기화된 count 버퍼: {initialized_buffers}")
            
            if missing_keys:
                print(f"   - 누락된 키 예시: {list(missing_keys)[:5]}")
            if unexpected_keys:
                print(f"   - 예상치 못한 키 예시: {list(unexpected_keys)[:5]}")
                
            
            print("✅ 모델 가중치 로드 완료 (레거시 호환 모드)")
            
        except Exception as e:
            raise RuntimeError(f"모델 가중치 로드 실패: {str(e)}") from e
        
        # 디바이스로 이동 및 평가 모드
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 버퍼 통계 복원 (expressibility 관련)
        if hasattr(self.model, 'prediction_head'):
            exp_mean = getattr(self.model.prediction_head, 'exp_mean', None)
            exp_std = getattr(self.model.prediction_head, 'exp_std', None)
            if exp_mean is not None and exp_std is not None:
                print(f"✅ Expressibility 통계 복원됨: mean={exp_mean.item():.6f}, std={exp_std.item():.6f}")
        
        print(f"✅ 모델 로드 완료 - 디바이스: {self.device}")
        return self.model
    
    def _extract_config_params(self, checkpoint, detected_config) -> dict:
        """체크포인트에서 설정 파라미터 추출"""
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_data = checkpoint['config']
            if isinstance(config_data, dict):
                # 유효한 PropertyPredictionConfig 파라미터만 필터링
                valid_params = {}
                config_fields = set(UnifiedPropertyPredictionConfig.__dataclass_fields__.keys())
                for key, value in config_data.items():
                    if key in config_fields:
                        valid_params[key] = value
                
                # 감지된 아키텍처로 덮어쓰기
                valid_params.update(detected_config)
                return valid_params
            else:
                # config 객체인 경우
                valid_params = {}
                config_fields = set(UnifiedPropertyPredictionConfig.__dataclass_fields__.keys())
                for field in config_fields:
                    if hasattr(config_data, field):
                        valid_params[field] = getattr(config_data, field)
                valid_params.update(detected_config)
                return valid_params
        else:
            # 기본 설정 + 감지된 아키텍처
            return self._create_default_config_dict(detected_config)
    
    def _create_default_config_dict(self, detected_config: dict) -> dict:
        """기본 설정 딕셔너리 생성"""
        default_config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'max_qubits': 50,
            'property_dim': 3,
            'attention_mode': 'advanced',
            'prediction_head_hidden_dim': 64,
        }
        default_config.update(detected_config)
        return default_config
    
    def _extract_state_dict(self, checkpoint_data):
        """체크포인트에서 state_dict 추출"""
        if 'model_state_dict' in checkpoint_data:
            return checkpoint_data['model_state_dict']
        elif 'model' in checkpoint_data:
            return checkpoint_data['model']
        elif 'state_dict' in checkpoint_data:
            return checkpoint_data['state_dict']
        else:
            raise ValueError("체크포인트에서 state_dict를 찾을 수 없습니다")
    
    def _detect_architecture_from_checkpoint(self, checkpoint_data):
        """체크포인트에서 정확한 아키텍처 파라미터 추출"""
        detected = {}
        
        # 1순위: model_info에서 직접 추출 (가장 정확함)
        if 'model_info' in checkpoint_data:
            model_info = checkpoint_data['model_info']
            if 'config' in model_info:
                config_info = model_info['config']
                detected.update({
                    'd_model': config_info.get('d_model'),
                    'n_layers': config_info.get('n_layers'),
                    'n_heads': config_info.get('n_heads'),
                    'd_ff': config_info.get('d_ff'),
                    'dropout': config_info.get('dropout'),
                    'cross_attention_heads': config_info.get('cross_attention_heads')
                })
                print(f"✅ model_info에서 아키텍처 추출: {detected}")
                return detected
        
        # 2순위: config 딕셔너리에서 직접 추출
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            detected.update({
                'd_model': config.get('d_model'),
                'n_layers': config.get('n_layers'), 
                'n_heads': config.get('n_heads'),
                'd_ff': config.get('d_ff'),
                'dropout': config.get('dropout')
            })
            # None 값 제거
            detected = {k: v for k, v in detected.items() if v is not None}
            if detected:
                print(f"✅ config에서 아키텍처 추출: {detected}")
                return detected
        
        # 3순위: state_dict에서 추론 (최후의 수단)
        state_dict = self._extract_state_dict(checkpoint_data)
        
        # d_model 감지 (feature_extractor의 첫 번째 레이어에서)
        for key, tensor in state_dict.items():
            if 'prediction_head.feature_extractor.0.weight' in key:
                detected['d_model'] = tensor.shape[1]  # input dimension
                break
            elif 'gate_embedding.weight' in key:
                detected['d_model'] = tensor.shape[1]  # embedding dimension
                break
        
        # n_layers 감지 (transformer_layers 개수)
        layer_indices = set()
        for key in state_dict.keys():
            if 'transformer_layers.' in key:
                try:
                    layer_idx = int(key.split('transformer_layers.')[1].split('.')[0])
                    layer_indices.add(layer_idx)
                except (ValueError, IndexError):
                    continue
        
        if layer_indices:
            detected['n_layers'] = max(layer_indices) + 1
        
        # n_heads 감지 (MultiHeadAttention의 가중치에서)
        for key, tensor in state_dict.items():
            if 'self_attn.in_proj_weight' in key and detected.get('d_model'):
                d_model = detected['d_model']
                # in_proj_weight: [3*d_model, d_model] (query, key, value 결합)
                if tensor.shape[0] == 3 * d_model and tensor.shape[1] == d_model:
                    # n_heads는 일반적으로 d_model의 약수
                    for n_heads in [4, 6, 8, 12, 16, 32]:
                        if d_model % n_heads == 0:
                            detected['n_heads'] = n_heads
                            break
                break
        
        if detected:
            print(f"🔍 state_dict에서 아키텍처 추론: {detected}")
        else:
            print("⚠️ 아키텍처 감지 실패")
        
        return detected
    
    def _safe_load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 안전 로딩"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                raise ValueError(f"체크포인트가 딕셔너리가 아님: {type(checkpoint)}")
            return checkpoint
        except Exception as e:
            raise RuntimeError(f"체크포인트 파일 읽기 실패: {checkpoint_path}") from e
    
    def get_model(self) -> Optional[UnifiedPropertyPredictionTransformer]:
        """로드된 모델 반환"""
        return self.model
    
    
    def get_config(self) -> Optional[UnifiedPropertyPredictionConfig]:
        """모델 설정 반환"""
        return self.config


def load_property_predictor(checkpoint_path: str = None, device: str = "auto") -> UnifiedPropertyPredictionTransformer:
    """
    체크포인트에서 자동으로 모델 구성을 감지하여 로드하는 헬퍼 함수
    
    Args:
        checkpoint_path: 체크포인트 파일 경로 (없을 경우 기본 경로에서 검색)
        device: 디바이스 설정 ("auto", "cuda", "cpu")
        
    Returns:
        로드된 모델 (PropertyPredictionTransformer 혹은 IntegratedPropertyPredictionTransformer)
    """
    loader = PropertyPredictorLoader(checkpoint_path, device)
    model = loader.load_model()
    
    print("\n✅ 모델 자동 로드 완료!")
    return model


def find_best_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    최적의 체크포인트 파일을 자동으로 찾는 함수
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 (기본값: "checkpoints")
        
    Returns:
        최적 체크포인트 파일 경로 또는 None
    """
    # 디렉토리 경로 저장
    checkpoint_path = Path(checkpoint_dir)
    
    # 디렉토리가 존재하지 않는 경우 상위 경로로 확장
    if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
        parent_dirs = [Path("."), Path("..")] 
        for parent in parent_dirs:
            alt_path = parent / checkpoint_path
            if alt_path.exists():
                checkpoint_path = alt_path
                print(f"🔎 체크포인트 디렉토리 발견: {alt_path}")
                break
    
    if not checkpoint_path.exists():
        print(f"⚠️ 체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
        return None
    
    # 우선순위 기반 체크포인트 파일 타입
    priority_filenames = [
        "best_model.pt",          # 최상위 우선순위 (현재 아키텍처)
        "best_enhanced_model.pt", # 레거시
        "best_property_model.pt", # 다음 우선순위
        "final_model.pt",         # 다음 우선순위
        "latest_model.pt"         # 다음 우선순위
    ]
    
    # 우선순위 리스트에서 찾기 
    for filename in priority_filenames:
        candidate = checkpoint_path / filename
        if candidate.exists():
            print(f"🏅 우선순위 체크포인트 발견: {candidate}")
            return str(candidate)
    
    # 패턴 기반 검색 (다양한 체크포인트 형태 고려)
    pattern_searches = [
        "*best*.pt",       # best로 시작하는 모든 파일
        "checkpoint*.pt", # checkpoint로 시작하는 모든 파일
        "model_*.pt",     # model_로 시작하는 모든 파일
        "*.pt"           # 모든 .pt 파일 (마지막 선택사항)
    ]
    
    # 각 패턴으로 검색
    for pattern in pattern_searches:
        matching_files = list(checkpoint_path.glob(pattern))
        if matching_files:
            # 수정 시간 기준 정렬
            latest = max(matching_files, key=lambda p: p.stat().st_mtime)
            print(f"📅 최신 체크포인트 발견 ({pattern}): {latest}")
            return str(latest)
    
    # 하위 디렉토리를 포함한 검색 (전체 리커시브 검색)
    all_pt_files = []
    for root, _, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith(".pt"):
                all_pt_files.append(Path(root) / file)
    
    if all_pt_files:
        latest = max(all_pt_files, key=lambda p: p.stat().st_mtime)
        print(f"🔍 하위 디렉토리 포함 검색 결과: {latest}")
        return str(latest)
    
    print("❌ 체크포인트 파일을 찾을 수 없습니다.")
    return None


if __name__ == "__main__":
    # 테스트 코드
    checkpoint_path = find_best_checkpoint()
    model = load_property_predictor(checkpoint_path)
    print(f"✅ 모델 로드 테스트 성공: {type(model).__name__}")
