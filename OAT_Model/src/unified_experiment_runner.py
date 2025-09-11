"""
Unified Experiment Runner
새로운 GNN 기반 임베딩 파이프라인을 지원하는 통합 실험 실행기
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import wandb

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

# Import unified config
from config.unified_training_config import (
    UnifiedTrainingConfig, 
    ConfigManager,
    get_config_by_name
)

# Import new encoding system
from encoding.encoding_pipeline_factory import EncodingPipelineFactory
from encoding.modular_quantum_attention import AttentionMode
from encoding.property_prediction_integration import (
    EnhancedPropertyPredictionEncoder,
    create_integrated_property_predictor,
    DEFAULT_CONFIG
)

# Import models
from models.decision_transformer import DecisionTransformer
from models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
from config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig

# Import data
from data.quantum_circuit_dataset import DatasetManager, create_dataloaders
from training.dataset.property_prediction_dataset import PropertyPredictionDataset,collate_fn
from training.utils.early_stopping import EarlyStopping
from training.utils.checkpoint_manager import CheckpointManager
import torch.nn.functional as F
# Import gates
from gates import QuantumGateRegistry

class UnifiedExperimentRunner:
    """새로운 임베딩 파이프라인을 지원하는 통합 실험 실행기"""
    
    def __init__(self, config: UnifiedTrainingConfig, model_type: str = "property_predictor"):
        self.config = config
        self.model_type = model_type

        self.loss_fn = F.mse_loss
        
        # GPU 최적화: 디바이스 설정
        if hasattr(config.model, 'get_device'):
            device_str = config.model.get_device()
        else:
            # Fallback: 기본 GPU 사용
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device_str)
        
        # CUDA 메모리 할당 전략 설정 (메모리 문제 해결)
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80%만 사용
            torch.backends.cudnn.benchmark = False  # 메모리 안정성을 위해 비활성화
            torch.backends.cudnn.deterministic = True
        
        # 그래디언트 축적(Gradient Accumulation) 사용 설정 - 메모리 절약을 위해 증가
        self.grad_accum_steps = getattr(self.config.training, 'grad_accum_steps', 8)  # 4에서 8로 증가
        # 배치 크기를 줄여서 메모리 사용량 감소
        original_batch_size = self.config.training.train_batch_size
        self.config.training.train_batch_size = original_batch_size
        self.effective_batch_size = self.config.training.train_batch_size * self.grad_accum_steps
        print(f"🔥 Gradient Accumulation: {self.grad_accum_steps} steps")
        print(f"🔥 Effective Batch Size: {self.effective_batch_size}")
        
        # GPU 메모리 최적화
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            
            # 메모리 활용 정보 표시
            reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
            allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"💾 Reserved Memory: {reserved_mem:.2f}GB | Allocated Memory: {allocated_mem:.2f}GB")
        
        print(f"🚀 Initializing Enhanced {model_type} Runner")
        print(f"Device: {self.device}")
        print(f"Experiment: {config.experiment.experiment_name}")
        print(f"Embedding Mode: {getattr(config.model, 'embedding_mode', 'gnn')}")
        print(f"Attention Mode: {getattr(config.model, 'attention_mode', 'advanced')}")
        
        # 디버깅 플래그
        self.debug_memory = False
        
        # wandb 초기화
        wandb.init(
            project="quantum-circuit-property-prediction",
            name=config.experiment.experiment_name,
            config={
                "model_type": model_type,
                "embedding_mode": getattr(config.model, 'embedding_mode', 'gnn'),
                "attention_mode": getattr(config.model, 'attention_mode', 'advanced'),
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.train_batch_size,
                "d_model": config.model.d_model,
                "n_heads": config.model.n_heads,
                "n_layers": config.model.n_layers,
                "entanglement_weight": getattr(config.training, 'entanglement_weight', 10.0),
                "expressibility_weight": getattr(config.training, 'expressibility_weight', 1.0),
                "fidelity_weight": getattr(config.training, 'fidelity_weight', 0.1),
            }
        )
        
        # Setup directories and seed
        config.setup_directories()
        config.set_seed()
        
        # Initialize encoding pipeline factory
        self.encoding_factory = EncodingPipelineFactory()
        
        # Initialize early stopping and checkpoint manager
        self.early_stopping = EarlyStopping(
            patience=getattr(config.training, 'early_stopping_patience', 15),
            min_delta=getattr(config.training, 'early_stopping_delta', 0.001)
        )
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.experiment.checkpoint_dir,
            device=self.device
        )
        
        # Initialize model with new embedding pipeline
        self.model = self._create_enhanced_model()
        
        # GPU 최적화: 모델을 GPU로 이동 및 최적화
        self.model.to(self.device)
        
        # 근본적 해결: 모든 서브모듈을 강제로 디바이스로 이동
        self._ensure_all_modules_on_device()
            
        # AMP (Automatic Mixed Precision) 설정 - 메모리 문제로 인해 비활성화
        self.use_amp = False  # getattr(config.model, 'use_amp', True) and self.device.type == 'cuda'
        # 최신 PyTorch 버전에 맞게 GradScaler 사용
        if self.use_amp:
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                # 이전 버전 호환성 유지
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        print(f"🚀 AMP Enabled: {self.use_amp}")
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _ensure_all_modules_on_device(self):
        """근본적 해결: 모든 서브모듈을 강제로 디바이스로 이동"""
        print(f"🔧 Ensuring all modules are on {self.device}")
        
        # 모든 서브모듈 순회하여 디바이스 이동
        for name, module in self.model.named_modules():
            if hasattr(module, 'to'):
                module.to(self.device)
                print(f"  ✅ Moved {name} to {self.device}")
        
        # 임베딩 파이프라인 특별 처리
        if hasattr(self.model, 'embedding_pipeline'):
            if hasattr(self.model.embedding_pipeline, 'unified_facade'):
                self.model.embedding_pipeline.unified_facade.to(self.device)
                print(f"  ✅ Moved embedding_pipeline.unified_facade to {self.device}")
            
            # 기타 임베딩 관련 모듈들
            for attr_name in ['grid_encoder', 'circuit_processor', 'batch_processor']:
                if hasattr(self.model.embedding_pipeline, attr_name):
                    attr_obj = getattr(self.model.embedding_pipeline, attr_name)
                    if hasattr(attr_obj, 'to'):
                        attr_obj.to(self.device)
                        print(f"  ✅ Moved embedding_pipeline.{attr_name} to {self.device}")
        
        print(f"🎯 All modules moved to {self.device}")
        
    def _create_enhanced_model(self):
        """Create model with enhanced embedding pipeline - SOTA 통합 모델 지원"""
        if self.model_type == "property_predictor":
            # Create unified model with size-based configuration
            from config.experiment_configs import MODEL_SIZES
            
            # 모델 크기 결정 (설정에서 지정하거나 기본값 사용)
            model_size = getattr(self.config.model, 'model_size', 'medium')
            if model_size not in MODEL_SIZES:
                print(f"⚠️ 알 수 없는 모델 크기: {model_size}, 기본값 'medium' 사용")
                model_size = 'medium'
            
            size_config = MODEL_SIZES[model_size]
            print(f"📏 모델 크기: {model_size} - d_model={size_config['d_model']}, n_layers={size_config['n_layers']}")
            
            # 통합된 설정으로 모델 생성
            config_params = {
                'd_model': size_config['d_model'],
                'n_heads': size_config['n_heads'], 
                'n_layers': size_config['n_layers'],
                'd_ff': size_config['d_ff'],
                'dropout': size_config['dropout'],
                'max_qubits': getattr(self.config.model, 'max_qubits', 10),
                'max_gates': getattr(self.config.model, 'max_gates', 100),
                'attention_mode': getattr(self.config.model, 'attention_mode', 'advanced'),
                'use_rotary_pe': getattr(self.config.model, 'use_rotary_pe', True),
                'cross_attention_heads': getattr(self.config.model, 'cross_attention_heads', 4),
                'consistency_loss_weight': getattr(self.config.model, 'consistency_loss_weight', 0.1),
                'gradient_clipping': getattr(self.config.model, 'gradient_clipping', 1.0),
                'numerical_stability': getattr(self.config.model, 'numerical_stability', True),
                'learning_rate': getattr(self.config.training, 'learning_rate', 1e-4),
                'train_batch_size': getattr(self.config.training, 'train_batch_size', size_config['batch_size']),
                'val_batch_size': getattr(self.config.training, 'val_batch_size', size_config['batch_size'] * 2)
            }
            
            integrated_config = UnifiedPropertyPredictionConfig(**config_params)
            model = UnifiedPropertyPredictionTransformer(integrated_config)
            print("🚀 SOTA 통합 모델 생성 완료")
            
            # GPU 메모리 최적화
            if self.device.type == 'cuda':
                model.half() if getattr(self.config.model, 'use_fp16', False) else model.float()
            
            return model
        elif self.model_type == "decision_transformer":
            # Create Decision Transformer with enhanced pipeline
            from models.decision_transformer import DecisionTransformer
            
            model_config = self.config.get_model_config_for_decision_transformer()
            model_config['device'] = self.device.type  # GPU 설정 추가
            model = DecisionTransformer(**model_config)
            
            # GPU 메모리 최적화
            if self.device.type == 'cuda':
                model.half() if getattr(self.config.model, 'use_fp16', False) else model.float()
            
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_optimizer(self):
        """Create optimizer - GPU 최적화"""
        # GPU에서 더 효율적인 AdamW 사용
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=1e-8,  # GPU 안정성
            amsgrad=True  # GPU에서 더 안정적
        )
        
        # GPU 메모리 최적화
        if self.device.type == 'cuda':
            # 옵티마이저 상태를 GPU로 이동
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        return optimizer
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        if self.config.training.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        else:
            return None
    
    def _create_loss_function(self):
        """손실 함수 생성 - SOTA 통합 모델 지원"""
        if self.model_type == "decision_transformer":
            return nn.CrossEntropyLoss()
        elif self.model_type == "property_predictor":
            # SOTA 모델인 경우 내장 손실 함수 사용, 아니면 MSE 사용
            use_sota = getattr(self.config.model, 'use_sota_architecture', True)
            if use_sota:
                return None  # SOTA 모델은 내장 손실 함수 사용
            else:
                return nn.MSELoss()
        else:
            return nn.MSELoss()
    
    def _prepare_batch_data(self, batch):
        """양자회로 그래프 구조를 보존하면서 메모리 효율적인 배치 데이터 준비"""
        
        if self.model_type == "property_predictor":
            circuit_specs = batch.get('circuit_specs', [])
            
            if not circuit_specs:
                raise ValueError("Empty circuit_specs provided - cannot create valid batch")
            
            # Extract target properties from batch['targets'] created by property_prediction collate_fn
            targets_dict = batch.get('targets')
            
            if targets_dict:
                # Convert dict format to tensor format [entanglement, expressibility, fidelity]
                batch_size = len(circuit_specs)
                target_properties = torch.zeros(batch_size, 3, device=self.device)
                
                # Extract values from targets dict
                if 'entanglement' not in targets_dict or 'expressibility' not in targets_dict or 'fidelity' not in targets_dict:
                    raise ValueError(f"Missing required target properties: {list(targets_dict.keys())}")
                
                entanglement_vals = targets_dict['entanglement']
                expressibility_vals = targets_dict['expressibility']
                fidelity_vals = targets_dict['fidelity']
                
                # Stack into [batch_size, 3] tensor
                target_properties[:, 0] = entanglement_vals.to(self.device)
                target_properties[:, 1] = expressibility_vals.to(self.device)
                target_properties[:, 2] = fidelity_vals.to(self.device)

            # 양자회로 그래프 데이터를 메모리 효율적으로 생성
            circuit_data_list = []
            
            # 각 circuit_spec에 대해 그래프 데이터 생성 (메모리 관리 강화)
            with torch.no_grad():  # 그래디언트 추적 비활성화
                for spec in circuit_specs:
                    if hasattr(spec, 'gates') and hasattr(spec, 'num_qubits'):
                        # 양자회로 그래프 빌더 사용 (본질 보존)
                        graph_builder = self.encoding_factory.graph_builder
                        graph_data = graph_builder.build_graph_from_circuit_spec(spec)
                        
                        # 즉시 GPU로 이동하여 메모리 효율성 향상
                        circuit_data = {
                            'node_features': graph_data.node_features.to(self.device, non_blocking=True),
                            'grid_positions': graph_data.grid_positions.to(self.device, non_blocking=True),
                            'node_types': graph_data.node_types.to(self.device, non_blocking=True),
                            'attention_mask': None
                        }
                        circuit_data_list.append(circuit_data)
                        
                        # 중간 객체 즉시 정리
                        del graph_data
            
            # 배치 텐서로 변환 (메모리 효율적 패딩)
            if circuit_data_list:
                batch_circuit_data = self._create_padded_batch_efficient(circuit_data_list)
            else:
                raise ValueError("No valid circuit data found in batch")
            
            return {
                'circuit_data': batch_circuit_data,
                'target_properties': target_properties
            }
            
        elif self.model_type == "decision_transformer":
            # Decision Transformer용 데이터 준비 (향후 구현)
            return batch
        else:
            return batch
    
    def _create_padded_batch_efficient(self, circuit_data_list):
        """메모리 효율적인 패딩 배치 생성"""
        batch_size = len(circuit_data_list)
        
        # 최대 노드 수 계산
        max_nodes = max(cd['node_features'].shape[0] for cd in circuit_data_list)
        max_nodes = min(max_nodes, 500)  # 메모리 안전을 위해 제한
        
        # 미리 할당된 배치 텐서
        batch_node_features = torch.zeros(batch_size, max_nodes, 12, device=self.device)
        batch_grid_positions = torch.zeros(batch_size, max_nodes, 2, device=self.device)
        batch_node_types = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=self.device)
        batch_attention_masks = torch.zeros(batch_size, max_nodes, device=self.device)
        
        # 각 샘플에 대해 효율적인 패딩
        for i, cd in enumerate(circuit_data_list):
            num_nodes = cd['node_features'].shape[0]
            actual_nodes = min(num_nodes, max_nodes)
            
            # 직접 복사 (효율적)
            batch_node_features[i, :actual_nodes] = cd['node_features'][:actual_nodes]
            batch_grid_positions[i, :actual_nodes] = cd['grid_positions'][:actual_nodes]
            batch_node_types[i, :actual_nodes] = cd['node_types'][:actual_nodes]
            
            # attention mask 설정
            batch_attention_masks[i, :actual_nodes] = 1.0
            
            # 중간 데이터 정리
            del cd['node_features']
            del cd['grid_positions'] 
            del cd['node_types']
        
        return {
            'node_features': batch_node_features,
            'grid_positions': batch_grid_positions,
            'node_types': batch_node_types,
            'attention_mask': batch_attention_masks
        }
    
    def train_epoch(self, train_loader):
        """한 에포크 학습 - 그래디언트 축적과 메모리 최적화"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 메모리 최적화: 가비지 콜렉션 강제 실행
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 메모리 정보 출력 (디버깅)
        if hasattr(self, 'debug_memory') and self.debug_memory:
            reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
            allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n💾 학습 시작 메모리: {reserved_mem:.2f}GB (예약) | {allocated_mem:.2f}GB (할당)")
        
        # 그래디언트 축적 설정
        grad_accum_steps = getattr(self, 'grad_accum_steps', 4)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad(set_to_none=True)  # 한번만 초기화 (메모리 효율)
        accum_loss = 0.0  # 축적된 손실
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 배치 데이터 준비
                prepared_batch = self._prepare_batch_data(batch)
                
                # 리소스 제한을 위해 배치를 하위 배치로 분할 (추가 최적화)
                sub_batch_factor = getattr(self.config.training, 'sub_batch_factor', 1)
                if sub_batch_factor > 1:
                    # TODO: 하위 배치 분할 로직 추가 (필요시)
                    pass
                
                # 메모리 사용량 디버깅 - 배치 진입 전
                if hasattr(self, 'debug_memory') and self.debug_memory and batch_idx < 30:
                    torch.cuda.synchronize()
                    reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
                    allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"\n💾 배치 {batch_idx+1} 시작 메모리: {reserved_mem:.2f}GB (예약) | {allocated_mem:.2f}GB (할당)")
                    
                    # 메모리 경계치 감지 및 클리어 시도 - 더 보수적으로 설정
                    if allocated_mem > 15.0:  # 20GB GPU에서 15GB이상이면 위험
                        print(f"\n⚠️ 메모리 경계치 도달: {allocated_mem:.2f}GB/20GB - 캐시 클리어 시도...")
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Forward pass with memory management
                # 그래프 작업 전 메모리 초기화
                if batch_idx % 10 == 0:
                    # 10배치마다 강제 메모리 정리
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Forward pass with AMP
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        if self.model_type == "property_predictor":
                            loss = self._train_step_property_predictor(prepared_batch)
                        elif self.model_type == "decision_transformer":
                            loss = self._train_step_decision_transformer(prepared_batch)
                        else:
                            continue
                else:
                    if self.model_type == "property_predictor":
                        loss = self._train_step_property_predictor(prepared_batch)
                    elif self.model_type == "decision_transformer":
                        loss = self._train_step_decision_transformer(prepared_batch)
                    else:
                        continue
                        
                # 메모리 문제 감지 (forward 후 체크)
                if hasattr(self, 'debug_memory') and self.debug_memory and batch_idx < 30:
                    torch.cuda.synchronize()
                    allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                    if allocated_mem > 17.0:  # 20GB GPU에서 17GB 이상이면 위험
                        print(f"\n⚠️ Forward 후 메모리 경계치 접근: {allocated_mem:.2f}GB/20GB")
                
                # Loss 유효성 검사
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # 그래디언트 축적을 위해 loss를 축적 스텝 수로 나눔
                scaled_loss = loss / grad_accum_steps
                
                # Backward pass with gradient accumulation
                if self.use_amp:
                    # AMP를 사용한 backward pass
                    self.scaler.scale(scaled_loss).backward()
                    accum_loss += loss.item()  # 원래 loss 값 추적
                    
                    # 메모리 체크 - backward 후
                    if hasattr(self, 'debug_memory') and self.debug_memory and batch_idx < 30:
                        torch.cuda.synchronize()  # 비동기 연산 완료 대기
                        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                        if allocated_mem > 17.5:  # 경계 상태 확인
                            print(f"\n⚠️ Backward 후 메모리 상태: {allocated_mem:.2f}GB/20GB")
                    
                    # 추적 단계가 다 차거나 마지막 배치일 경우
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                        # AMP에서는 항상 unscale_을 먼저 호출해야 함
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping
                        if self.config.training.gradient_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.training.gradient_clip_norm
                            )
                        
                        # 연산 추적 및 매개변수 업데이트
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율적 초기화
                        
                        # 디버깅 메모리 정보 - 옵티마이저 스텝 후
                        if hasattr(self, 'debug_memory') and self.debug_memory:
                            torch.cuda.synchronize()  # 비동기 연산 완료 대기
                            reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
                            allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                            print(f"\n💾 배치 {batch_idx+1} 업데이트 후 메모리: {reserved_mem:.2f}GB (예약) | {allocated_mem:.2f}GB (할당)")
                            
                            # 메모리 상태가 위험 수준이면 강제 정리
                            if allocated_mem > 16.0 or batch_idx % 10 == 0:
                                print(f"\n🧹 메모리 정리 시도 (배치 {batch_idx+1})")
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()  # 정리 완료 대기
                                
                                # 정리 후 메모리 상태 확인
                                new_mem = torch.cuda.memory_allocated(0) / 1024**3
                                reduction = allocated_mem - new_mem
                                print(f"   → 메모리 {reduction:.2f}GB 정리됨 (현재: {new_mem:.2f}GB)")
                else:
                    # 일반 backward pass
                    scaled_loss.backward()
                    accum_loss += loss.item()
                    
                    # 추적 단계가 다 차거나 마지막 배치일 경우
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                        # Gradient clipping
                        if self.config.training.gradient_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.training.gradient_clip_norm
                            )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                
                # Update metrics - loss.item()으로 실제 값 추출
                current_loss = loss.item()
                total_loss += current_loss
                self.current_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
                # 메모리 최적화: 더 자주 캐시 정리 (CUDA 메모리 오류 방지)
                if batch_idx % 5 == 0:  # 10에서 5로 변경하여 더 자주 정리
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()  # 동기화 추가
                
                # 배치 완료 후 즉시 메모리 정리
                del prepared_batch
                if 'circuit_data' in locals():
                    del circuit_data
                if 'target_properties' in locals():
                    del target_properties
                torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    print(f"\nCUDA memory error at batch {batch_idx}: {e}")
                    print("Clearing cache and continuing...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # 최종 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _train_step_property_predictor(self, prepared_batch):
        """Property Predictor 학습 스텝 - SOTA 통합 모델 지원"""
        circuit_data = prepared_batch['circuit_data']
        target_properties = prepared_batch['target_properties']
        
        # 디바이스로 이동
        for key, value in circuit_data.items():
            if isinstance(value, torch.Tensor):
                circuit_data[key] = value.to(self.device)
        
        if target_properties is not None:
            target_properties = target_properties.to(self.device)
        
        # SOTA 통합 모델 여부 확인
        use_sota = getattr(self.config.model, 'use_sota_architecture', True)
        
        if use_sota and isinstance(self.model, UnifiedPropertyPredictionTransformer):
            # SOTA 통합 모델 사용
            outputs = self.model(circuit_data)
            
            # 타겟을 딕셔너리 형태로 변환
            if target_properties is not None:
                targets_dict = {
                    'entanglement': target_properties[:, 0],
                    'fidelity': target_properties[:, 2],  # fidelity가 2번째 인덱스
                    'expressibility': target_properties[:, 1]  # expressibility가 1번째 인덱스
                }
                
                # SOTA 모델의 내장 손실 함수 사용
                losses = self.model.compute_loss(outputs, targets_dict)
                
                # 손실 컴포넌트 저장
                self._last_loss_components = {
                    'entanglement_loss': losses.get('entanglement', torch.tensor(0.0)).item(),
                    'expressibility_loss': losses.get('expressibility', torch.tensor(0.0)).item(),
                    'fidelity_loss': losses.get('fidelity', torch.tensor(0.0)).item(),
                    'consistency_loss': losses.get('consistency', torch.tensor(0.0)).item(),
                    'total_loss': losses['total'].item()
                }
                
                return losses['total']
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # 기존 모델 로직
            outputs = self.model(circuit_data, mode='unified')
        
        # # 디버깅: 모델 출력 키 확인
        # print(f"Model output keys: {list(outputs.keys())}")
        # if 'unified' in outputs:
        #     print(f"Unified keys: {list(outputs['unified'].keys())}")
        
        # 손실 계산 - 개별 property별 가중 손실
        if 'unified' in outputs:
            entanglement_pred = outputs['unified']['entanglement']
            expressibility_pred = outputs['unified']['expressibility'] 
            fidelity_pred = outputs['unified']['fidelity']
        else:
            # Property predictor는 직접 property 키를 반환
            entanglement_pred = outputs.get('entanglement', torch.zeros(1, 1, device=self.device))
            expressibility_pred = outputs.get('expressibility', torch.zeros(1, 1, device=self.device))
            fidelity_pred = outputs.get('fidelity', torch.zeros(1, 1, device=self.device))
        if target_properties is not None:
            # NaN 검사 및 디버깅
            has_nan = (torch.isnan(entanglement_pred).any() or torch.isnan(expressibility_pred).any() or 
                      torch.isnan(fidelity_pred).any() or torch.isnan(target_properties).any())
            
            if has_nan:
                print(f"\n⚠️ NaN 감지! 예측과 타겟 디버깅:")
                print(f"  entanglement_pred NaN: {torch.isnan(entanglement_pred).any()}")
                print(f"  expressibility_pred NaN: {torch.isnan(expressibility_pred).any()}")
                print(f"  fidelity_pred NaN: {torch.isnan(fidelity_pred).any()}")
                print(f"  target_properties NaN: {torch.isnan(target_properties).any()}")
                print(f"  entanglement_pred range: {entanglement_pred.min().item():.6f} - {entanglement_pred.max().item():.6f}")
                print(f"  expressibility_pred range: {expressibility_pred.min().item():.6f} - {expressibility_pred.max().item():.6f}")
                print(f"  fidelity_pred range: {fidelity_pred.min().item():.6f} - {fidelity_pred.max().item():.6f}")
                
                # NaN 발생 시 훈련 중단
                raise RuntimeError("NaN detected in predictions or targets - training cannot continue")
            
            # 타겟을 개별 property로 분할 (entanglement, expressibility, fidelity 순서)
            target_entanglement = target_properties[:, 0:1]
            target_expressibility = target_properties[:, 1:2] 
            target_fidelity = target_properties[:, 2:3]
            
            # 개별 property 손실 계산
            entanglement_loss = self.loss_fn(entanglement_pred, target_entanglement)
            expressibility_loss = self.loss_fn(expressibility_pred, target_expressibility)
            fidelity_loss = self.loss_fn(fidelity_pred, target_fidelity)
            
            # 가중치 적용 (config에서 가져오기)
            entanglement_weight = getattr(self.config.training, 'entanglement_weight', 10.0)
            expressibility_weight = getattr(self.config.training, 'expressibility_weight', 1.0)
            fidelity_weight = getattr(self.config.training, 'fidelity_weight', 0.1)
            
            # 가중 손실 계산
            weighted_entanglement_loss = entanglement_weight * entanglement_loss
            weighted_expressibility_loss = expressibility_weight * expressibility_loss
            weighted_fidelity_loss = fidelity_weight * fidelity_loss
            
            # 총 손실
            total_loss = weighted_entanglement_loss + weighted_expressibility_loss + weighted_fidelity_loss
            #print(f"total_loss: {total_loss}")
            # 로스 컴포넌트 저장 (wandb 로깅용)
            self._last_loss_components = {
                'entanglement_loss': entanglement_loss.item(),
                'expressibility_loss': expressibility_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'weighted_entanglement_loss': weighted_entanglement_loss.item(),
                'weighted_expressibility_loss': weighted_expressibility_loss.item(),
                'weighted_fidelity_loss': weighted_fidelity_loss.item()
            }
            
            # wandb 로깅 및 디버깅 정보
            if hasattr(self, 'current_step'):
                # 주기적으로 타겟 값 범위 로깅 (100스텝마다)
                if self.current_step % 100 == 0:
                    wandb.log({
                        'debug/target_entanglement_mean': target_entanglement.mean().item(),
                        'debug/target_entanglement_std': target_entanglement.std().item(),
                        'debug/target_expressibility_mean': target_expressibility.mean().item(),
                        'debug/target_expressibility_std': target_expressibility.std().item(),
                        'debug/target_fidelity_mean': target_fidelity.mean().item(),
                        'debug/target_fidelity_std': target_fidelity.std().item(),
                        'debug/pred_entanglement_mean': entanglement_pred.mean().item(),
                        'debug/pred_expressibility_mean': expressibility_pred.mean().item(),
                        'debug/pred_fidelity_mean': fidelity_pred.mean().item(),
                        'step': self.current_step
                    })
                    
                # 50배치마다 MAE 로깅
                if self.current_step % 50 == 0:
                    # MAE 계산 (Mean Absolute Error)
                    entanglement_mae = torch.nn.functional.l1_loss(entanglement_pred, target_entanglement).item()
                wandb.log({
                    'train/entanglement_loss': entanglement_loss.item(),
                    'train/expressibility_loss': expressibility_loss.item(),
                    'train/fidelity_loss': fidelity_loss.item(),
                    'train/weighted_entanglement_loss': weighted_entanglement_loss.item(),
                    'train/weighted_expressibility_loss': weighted_expressibility_loss.item(),
                    'train/weighted_fidelity_loss': weighted_fidelity_loss.item(),
                    'train/total_weighted_loss': total_loss.item(),
                    'train/entanglement_weight': entanglement_weight,
                    'train/expressibility_weight': expressibility_weight,
                    'train/fidelity_weight': fidelity_weight,
                    'step': self.current_step
                })

            loss = total_loss

            # 손실 값 검증
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Invalid loss detected: {loss.item()}")
            return loss
        else:
            print("No unified output")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def validate(self, val_loader):
        """검증 스텝 수행"""
        self.model.eval()
        total_losses = {'total_loss': 0.0, 'entanglement_loss': 0.0, 'expressibility_loss': 0.0, 'fidelity_loss': 0.0,
                      'weighted_entanglement_loss': 0.0, 'weighted_expressibility_loss': 0.0, 'weighted_fidelity_loss': 0.0}
        val_predictions = []
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                prepared_batch = self._prepare_batch_data(batch)
                
                if self.model_type == "property_predictor":
                    losses = self._val_step_property_predictor(prepared_batch)
                    
                    # 예측값과 타겟값 저장 (MAE 계산용)
                    if 'entanglement_pred' in losses and 'entanglement_target' in losses:
                        val_predictions.append({
                            'entanglement_pred': losses['entanglement_pred'].detach(),
                            'expressibility_pred': losses['expressibility_pred'].detach(),
                            'fidelity_pred': losses['fidelity_pred'].detach(),
                            'entanglement_target': losses['entanglement_target'].detach(),
                            'expressibility_target': losses['expressibility_target'].detach(),
                            'fidelity_target': losses['fidelity_target'].detach()
                        })
                    
                    # 손실값 누적
                    for key, value in losses.items():
                        if key in total_losses:
                            if isinstance(value, torch.Tensor):
                                total_losses[key] += value.item()
                            elif isinstance(value, (int, float)):
                                total_losses[key] += value
                            
                elif self.model_type == "decision_transformer":
                    loss = self._train_step_decision_transformer(prepared_batch)
                    total_losses['total_loss'] += loss.item()
                else:
                    continue
        
        # 평균 계산 및 wandb 로깅
        avg_losses = {key: value / max(num_batches, 1) for key, value in total_losses.items()}
        
        # MAE 계산을 위한 예측값과 실제값 누적 (validation 중에 수집)
        all_entanglement_pred = torch.cat([item['entanglement_pred'] for item in val_predictions]) if val_predictions else torch.tensor([])
        all_expressibility_pred = torch.cat([item['expressibility_pred'] for item in val_predictions]) if val_predictions else torch.tensor([])
        all_fidelity_pred = torch.cat([item['fidelity_pred'] for item in val_predictions]) if val_predictions else torch.tensor([])
        
        all_entanglement_target = torch.cat([item['entanglement_target'] for item in val_predictions]) if val_predictions else torch.tensor([])
        all_expressibility_target = torch.cat([item['expressibility_target'] for item in val_predictions]) if val_predictions else torch.tensor([])
        all_fidelity_target = torch.cat([item['fidelity_target'] for item in val_predictions]) if val_predictions else torch.tensor([])
        
        # Validation MAE 계산
        val_mae = {}
        if len(all_entanglement_pred) > 0:
            val_mae['entanglement'] = self.loss_fn(all_entanglement_pred, all_entanglement_target).item()
            val_mae['expressibility'] = self.loss_fn(all_expressibility_pred, all_expressibility_target).item()
            val_mae['fidelity'] = self.loss_fn(all_fidelity_pred, all_fidelity_target).item()
            val_mae['total'] = (val_mae['entanglement'] + val_mae['expressibility'] + val_mae['fidelity']) / 3.0
        
        # 콘솔 출력 - 트레이닝과 동일한 형식으로 상세 로깅
        print(f"\n📊 Validation Results (Epoch {self.current_epoch}):")
        print(f"   Total Loss: {avg_losses['total_loss']:.6f}")
        print(f"   Entanglement Loss: {avg_losses.get('entanglement_loss', 0):.6f}")
        print(f"   Expressibility Loss: {avg_losses.get('expressibility_loss', 0):.6f}")
        print(f"   Fidelity Loss: {avg_losses.get('fidelity_loss', 0):.6f}")
        print(f"   Weighted Entanglement Loss: {avg_losses.get('weighted_entanglement_loss', 0):.6f}")
        print(f"   Weighted Expressibility Loss: {avg_losses.get('weighted_expressibility_loss', 0):.6f}")
        print(f"   Weighted Fidelity Loss: {avg_losses.get('weighted_fidelity_loss', 0):.6f}")
        
        if val_mae:
            print(f"   MAE - Entanglement: {val_mae['entanglement']:.6f}")
            print(f"   MAE - Expressibility: {val_mae['expressibility']:.6f}")
            print(f"   MAE - Fidelity: {val_mae['fidelity']:.6f}")
            print(f"   MAE - Total: {val_mae['total']:.6f}")
        
        # wandb 검증 로깅
        wandb_log_data = {
            'val/total_loss': avg_losses['total_loss'],
            'val/entanglement_loss': avg_losses.get('entanglement_loss', 0),
            'val/expressibility_loss': avg_losses.get('expressibility_loss', 0),
            'val/fidelity_loss': avg_losses.get('fidelity_loss', 0),
            'val/weighted_entanglement_loss': avg_losses.get('weighted_entanglement_loss', 0),
            'val/weighted_expressibility_loss': avg_losses.get('weighted_expressibility_loss', 0),
            'val/weighted_fidelity_loss': avg_losses.get('weighted_fidelity_loss', 0),
            'epoch': self.current_epoch
        }
        
        # MAE 로깅 추가
        if val_mae:
            wandb_log_data.update({
                'val_mae/entanglement': val_mae['entanglement'],
                'val_mae/expressibility': val_mae['expressibility'],
                'val_mae/fidelity': val_mae['fidelity'],
                'val_mae/total': val_mae['total']
            })
        
        wandb.log(wandb_log_data)
        
        return avg_losses['total_loss']
    
    def train_decision_transformer(self, train_loader, val_loader, rtg_calculator):
        """Decision Transformer 학습 메서드 - RTG 가이드 포함"""
        print(f"\n🚀 Starting Decision Transformer Training")
        print(f"   Epochs: {self.config.training.num_epochs}")
        print(f"   Learning Rate: {self.config.training.learning_rate}")
        print(f"   Batch Size: {self.config.training.train_batch_size}")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n📈 Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch_decision_transformer(train_loader, rtg_calculator)
            
            # Validation phase
            val_loss = self._validate_epoch_decision_transformer(val_loader, rtg_calculator)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = Path(self.config.experiment.checkpoint_dir) / "best_decision_transformer.pt"
                self.save_checkpoint(best_model_path)
                print(f"💾 New best model saved: {val_loss:.6f}")
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        print(f"\n✅ Decision Transformer training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
    
    def _train_epoch_decision_transformer(self, train_loader, rtg_calculator):
        """Decision Transformer 학습 에포크"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 배치가 이미 콜레이터에 의해 처리되었으므로 RTG도 포함되어 있음
                if isinstance(batch, dict) and 'rtg_rewards' in batch:
                    prepared_batch = batch
                else:
                    # RTG 계산 필수
                    if rtg_calculator is None:
                        raise ValueError("RTG calculator required for Decision Transformer training")
                    rtg_values = self._calculate_batch_rtg(batch, rtg_calculator)
                    prepared_batch = self._prepare_decision_transformer_batch(batch, rtg_values)
                
                # 근본적 해결 후 불필요한 수동 디바이스 이동 제거
                # (모든 모듈이 이미 올바른 디바이스에 있음)
                
                # Forward pass
                outputs = self.model(
                    input_sequence=prepared_batch['input_sequence'],
                    attention_mask=prepared_batch['attention_mask'],
                    action_prediction_mask=prepared_batch['action_prediction_mask'],
                    rtg_rewards=prepared_batch['rtg_rewards']
                )
                print(prepared_batch)
                # Loss 계산
                loss_dict = self.model.compute_loss(
                    predictions=outputs,
                    targets=prepared_batch['targets'],
                    action_prediction_mask=prepared_batch['action_prediction_mask']
                )
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                self.current_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch_decision_transformer(self, val_loader, rtg_calculator):
        """Decision Transformer 검증 에포크"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validation Epoch {self.current_epoch}")
            for batch_idx, batch in enumerate(pbar_val):
                try:
                    # 배치가 이미 콜레이터에 의해 처리되었으므로 RTG도 포함되어 있음
                    if isinstance(batch, dict) and 'rtg_rewards' in batch:
                        prepared_batch = batch
                    else:
                        # RTG 계산 필수
                        if rtg_calculator is None:
                            raise ValueError("RTG calculator required for Decision Transformer validation")
                        rtg_values = self._calculate_batch_rtg(batch, rtg_calculator)
                        prepared_batch = self._prepare_decision_transformer_batch(batch, rtg_values)
                    
                    # Forward pass
                    outputs = self.model(
                        input_sequence=prepared_batch['input_sequence'],
                        attention_mask=prepared_batch['attention_mask'],
                        action_prediction_mask=prepared_batch['action_prediction_mask'],
                        rtg_rewards=prepared_batch['rtg_rewards']
                    )
                    
                    # Loss 계산
                    loss_dict = self.model.compute_loss(
                        predictions=outputs,
                        targets=prepared_batch['targets'],
                        action_prediction_mask=prepared_batch['action_prediction_mask']
                    )
                    
                    loss = loss_dict['total_loss']
                    total_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        return total_loss / max(num_batches, 1)
    
    def _calculate_batch_rtg(self, batch, rtg_calculator):
        """배치에 대한 RTG 값 계산"""
        # 배치가 이미 콜레이터에 의해 처리된 경우, RTG는 이미 계산되어 있음
        if isinstance(batch, dict) and 'rtg_rewards' in batch:
            return batch['rtg_rewards']
        
        # 원시 CircuitData 객체들인 경우 (이 경우는 발생하지 않을 것임)
        rtg_values_list = []
        
        for circuit_data in batch:
            # CircuitData에서 CircuitSpec 추출
            circuit_spec = circuit_data.circuit_spec if hasattr(circuit_data, 'circuit_spec') else circuit_data
            
            # 타겟 속성 추출
            target_properties = {
                'entanglement': getattr(circuit_spec, 'target_entanglement', 0.8),
                'fidelity': getattr(circuit_spec, 'target_fidelity', 0.9),
                'expressibility': getattr(circuit_spec, 'target_expressibility', 0.7)
            }
            
            # RTG 계산 (단일 회로에 대해)
            rtg_values = rtg_calculator.calculate_rtg_sequence([circuit_spec], target_properties)
            rtg_values_list.append(rtg_values)
        
        return rtg_values_list
    
    def _prepare_decision_transformer_batch(self, batch, rtg_values_list):
        """Decision Transformer용 배치 준비 - 모델 내장 SAR 로직 활용"""
        debug_log("Preparing DT batch - delegating SAR creation to model")
        
        batch_size = len(batch)
        
        # 1. 입력 시퀀스 패딩 (모델이 SAR 변환 처리)
        input_sequences = [item['decision_transformer']['embeddings'] for item in batch]
        max_seq_len = max(seq.shape[1] for seq in input_sequences)
        embed_dim = input_sequences[0].shape[2]
        
        padded_input = torch.zeros(batch_size, max_seq_len, embed_dim, device=self.device)
        for i, seq in enumerate(input_sequences):
            seq_len = seq.shape[1]
            padded_input[i, :seq_len] = seq.squeeze(0).to(self.device)
        
        # 2. Attention mask 패딩
        attention_masks = [item['decision_transformer']['attention_mask'] for item in batch]
        padded_attention = torch.zeros(batch_size, max_seq_len, max_seq_len, dtype=torch.bool, device=self.device)
        for i, mask in enumerate(attention_masks):
            seq_len = mask.shape[1]
            padded_attention[i, :seq_len, :seq_len] = mask.squeeze(0).to(self.device)
        
        # 3. Action prediction mask 패딩
        action_masks = [item['decision_transformer']['action_prediction_mask'] for item in batch]
        padded_action_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=self.device)
        for i, mask in enumerate(action_masks):
            seq_len = mask.shape[1]
            padded_action_mask[i, :seq_len] = mask.squeeze(0).to(self.device)
        
        # 4. RTG 패딩
        padded_rtg = torch.zeros(batch_size, max_seq_len, device=self.device)
        for i, rtg in enumerate(rtg_values_list):
            if isinstance(rtg, torch.Tensor):
                rtg = rtg.to(self.device)
                rtg_len = min(rtg.shape[0], max_seq_len)
                padded_rtg[i, :rtg_len] = rtg[:rtg_len]
            else:
                padded_rtg[i, :] = float(rtg)
        
        # 5. Targets 패딩
        targets = []
        for item in batch:
            if 'decision_transformer' in item and 'targets' in item['decision_transformer']:
                targets.append(item['decision_transformer']['targets'])
            else:
                raise ValueError("Missing targets in decision transformer batch item")
        
        target_dim = targets[0].shape[2] if targets else 3
        padded_targets = torch.zeros(batch_size, max_seq_len, target_dim, device=self.device)
        for i, target in enumerate(targets):
            target = target.to(self.device)
            seq_len = min(target.shape[1], max_seq_len)
            padded_targets[i, :seq_len] = target.squeeze(0)[:seq_len]
        
        debug_log(f"DT batch - input: {padded_input.shape}, rtg: {padded_rtg.shape}, targets: {padded_targets.shape}")
        
        return {
            'input_sequence': padded_input,
            'attention_mask': padded_attention,
            'action_prediction_mask': padded_action_mask,
            'rtg_rewards': padded_rtg,
            'targets': padded_targets
        }
    
    def save_checkpoint(self, filepath):
        """체크포인트 저장"""
        # PropertyConfig 호환 설정만 저장
        property_config = {}
        if hasattr(self.config, 'model') and self.config.model:
            # PropertyConfig 필드만 추출
            property_fields = [
                'device', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'dropout',
                'attention_mode', 'use_rotary_pe', 'cross_attention_heads',
                'property_dim', 'max_qubits', 'max_gates', 'learning_rate',
                'min_learning_rate', 'train_batch_size', 'val_batch_size',
                'grad_accum_steps', 'warmup_steps', 'weight_decay',
                'consistency_loss_weight', 'numerical_stability', 'gradient_clipping'
            ]
            
            for field in property_fields:
                if hasattr(self.config.model, field):
                    property_config[field] = getattr(self.config.model, field)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_val_loss': self.best_val_loss,
            'config': property_config,  # PropertyConfig 호환 설정만 저장
            'model_type': self.model_type
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def train(self, train_loader, val_loader=None):
        """메인 학습 루프 - property_predictor는 모듈화된 트레이너 사용"""
        if self.model_type == "property_predictor":
            # Property Predictor는 모듈화된 트레이너에 완전 위임
            from training.property_prediction_trainer import create_trainer
            
            print(f"🎯 Starting modular training for {self.config.training.num_epochs} epochs")
            print(f"📊 Model Type: SOTA 통합 모델")
            
            # 모듈화된 트레이너 생성
            trainer = create_trainer(
                model=self.model,
                config=self.config,
                device=str(self.device),
                use_wandb=True
            )
            
            # 학습 실행 (완전히 트레이너에게 위임)
            results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.config.training.num_epochs
            )
            
            print("🎉 Modular training completed!")
            return results

    def train_step(self, batch):
        """Single training step with enhanced pipeline - GPU 최적화"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)  # GPU 메모리 최적화
        
        # GPU로 데이터 이동
        circuit_specs = batch['circuit_specs']
        targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                  for k, v in batch['targets'].items()}
        
        # AMP를 사용한 Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(circuit_specs)
                loss = self._compute_loss(outputs, targets)
        else:
            outputs = self.model(circuit_specs)
            loss = self._compute_loss(outputs, targets)
        
        # AMP를 사용한 Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # GPU 메모리 정리
        if self.device.type == 'cuda' and torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
            torch.cuda.empty_cache()
        
        return {
            'loss': loss.item(),
            'outputs': outputs
        }
    
    def _val_step_property_predictor(self, prepared_batch):
        """Property Predictor 검증 스텝 - 학습 스텝과 동일하지만 그래디언트 없음"""
        circuit_data = prepared_batch['circuit_data']
        target_properties = prepared_batch['target_properties']
        
        # 디바이스로 이동
        for key, value in circuit_data.items():
            if isinstance(value, torch.Tensor):
                circuit_data[key] = value.to(self.device)
        
        if target_properties is not None:
            target_properties = target_properties.to(self.device)
        
        # Forward pass
        outputs = self.model(circuit_data, mode='unified')
        
        # 손실 계산 - 개별 property별 가중 손실
        if 'unified' in outputs:
            entanglement_pred = outputs['unified']['entanglement']
            expressibility_pred = outputs['unified']['expressibility'] 
            fidelity_pred = outputs['unified']['fidelity']
        else:
            # Property predictor는 직접 property 키를 반환
            entanglement_pred = outputs.get('entanglement', torch.zeros(1, 1, device=self.device))
            expressibility_pred = outputs.get('expressibility', torch.zeros(1, 1, device=self.device))
            fidelity_pred = outputs.get('fidelity', torch.zeros(1, 1, device=self.device))
        
        if target_properties is not None:
            # NaN 검사 및 디버깅
            has_nan = (torch.isnan(entanglement_pred).any() or torch.isnan(expressibility_pred).any() or 
                      torch.isnan(fidelity_pred).any() or torch.isnan(target_properties).any())
            
            if has_nan:
                print(f"\n⚠️ Validation NaN 감지! 예측과 타겟 디버깅:")
                print(f"  entanglement_pred NaN: {torch.isnan(entanglement_pred).any()}")
                print(f"  expressibility_pred NaN: {torch.isnan(expressibility_pred).any()}")
                print(f"  fidelity_pred NaN: {torch.isnan(fidelity_pred).any()}")
                print(f"  target_properties NaN: {torch.isnan(target_properties).any()}")
                print(f"  entanglement_pred range: {entanglement_pred.min().item():.6f} - {entanglement_pred.max().item():.6f}")
                print(f"  expressibility_pred range: {expressibility_pred.min().item():.6f} - {expressibility_pred.max().item():.6f}")
                print(f"  fidelity_pred range: {fidelity_pred.min().item():.6f} - {fidelity_pred.max().item():.6f}")
                
                # NaN을 0.5로 대체하여 손실 계산 계속
                entanglement_pred = torch.nan_to_num(entanglement_pred, nan=0.5)
                expressibility_pred = torch.nan_to_num(expressibility_pred, nan=0.5)
                fidelity_pred = torch.nan_to_num(fidelity_pred, nan=0.5)
                target_properties = torch.nan_to_num(target_properties, nan=0.5)
                print(f"  NaN을 0.5로 대체하여 손실 계산 계속")
            
            # 타겟을 개별 property로 분할 (entanglement, expressibility, fidelity 순서)
            target_entanglement = target_properties[:, 0:1]
            target_expressibility = target_properties[:, 1:2] 
            target_fidelity = target_properties[:, 2:3]
            
            # 개별 property 손실 계산
            entanglement_loss = self.loss_fn(entanglement_pred, target_entanglement)
            expressibility_loss = self.loss_fn(expressibility_pred, target_expressibility)
            fidelity_loss = self.loss_fn(fidelity_pred, target_fidelity)
            
            # 가중치 적용 (config에서 가져오기)
            entanglement_weight = getattr(self.config.training, 'entanglement_weight', 10.0)
            expressibility_weight = getattr(self.config.training, 'expressibility_weight', 1.0)
            fidelity_weight = getattr(self.config.training, 'fidelity_weight', 0.1)
            
            # 가중 손실 계산
            weighted_entanglement_loss = entanglement_weight * entanglement_loss
            weighted_expressibility_loss = expressibility_weight * expressibility_loss
            weighted_fidelity_loss = fidelity_weight * fidelity_loss
            
            # 총 손실
            total_loss = weighted_entanglement_loss + weighted_expressibility_loss + weighted_fidelity_loss
            
            # 손실 값 검증
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise RuntimeError(f"Invalid validation loss detected: {total_loss.item()}")
            
            return {
                'total_loss': total_loss.item(),
                'entanglement_loss': entanglement_loss.item(),
                'expressibility_loss': expressibility_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'weighted_entanglement_loss': weighted_entanglement_loss.item(),
                'weighted_expressibility_loss': weighted_expressibility_loss.item(),
                'weighted_fidelity_loss': weighted_fidelity_loss.item(),
                'entanglement_pred': entanglement_pred,
                'expressibility_pred': expressibility_pred,
                'fidelity_pred': fidelity_pred,
                'entanglement_target': target_entanglement,
                'expressibility_target': target_expressibility,
                'fidelity_target': target_fidelity
            }
        else:
            print(f"⚠️ Warning: target_properties is None in validation batch - this should not happen!")
            print(f"   prepared_batch keys: {list(prepared_batch.keys())}")
            if 'target_properties' in prepared_batch:
                print(f"   target_properties type: {type(prepared_batch['target_properties'])}")
                print(f"   target_properties value: {prepared_batch['target_properties']}")
            
            # target이 없으면 validation을 수행할 수 없으므로 에러 발생
            raise ValueError("Validation batch missing target_properties - cannot compute validation loss")

    def create_dataloaders(self, dataset_manager):
        """Create optimized dataloaders - GPU 최적화"""
        train_dataset, val_dataset, test_dataset = dataset_manager.create_datasets()
        
        # GPU 최적화 DataLoader 설정
        dataloader_kwargs = {
            'pin_memory': getattr(self.config.model, 'pin_memory', True) and self.device.type == 'cuda',
            'num_workers': getattr(self.config.model, 'num_workers', 4),
            'prefetch_factor': getattr(self.config.model, 'prefetch_factor', 2),
            'persistent_workers': True if getattr(self.config.model, 'num_workers', 4) > 0 else False
        }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )
        
        print(f"🚀 DataLoader GPU optimizations: pin_memory={dataloader_kwargs['pin_memory']}, num_workers={dataloader_kwargs['num_workers']}")
        
        return train_loader, val_loader

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Unified Enhanced Experiment Runner")
    parser.add_argument("--config", type=str, default="medium", 
                       help="Configuration name (small, medium, large) or path to config file")
    parser.add_argument("--model", type=str, choices=["decision_transformer", "property_predictor"],
                       default="property_predictor", help="Model type to train")
    parser.add_argument("--embedding-mode", type=str, choices=["gnn", "hybrid"],
                       default="gnn", help="Embedding pipeline mode")
    parser.add_argument("--attention-mode", type=str, choices=["standard", "advanced"],
                       default="standard", help="Attention mechanism mode")
    parser.add_argument("--experiment-name", type=str, help="Custom experiment name")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    
    args = parser.parse_args()
    
    # Load or create configuration
    config_manager = ConfigManager()
    
    if Path(args.config).exists():
        config = UnifiedTrainingConfig.load(args.config)
    else:
        config = get_config_by_name(args.config)
    
    # Apply command line overrides
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    else:
        # 자동 실험명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment.experiment_name = f"enhanced_{args.model}_{args.embedding_mode}_{args.attention_mode}_{timestamp}"
    
    if args.data_path:
        config.data.data_path = args.data_path
    
    # 새로운 설정 추가
    config.model.embedding_mode = args.embedding_mode
    config.model.attention_mode = args.attention_mode
    config.model.model_size = args.config  # Command line config maps to model size
    
    # Save the configuration
    config_path = Path(config.experiment.output_dir) / f"{config.experiment.experiment_name}_config.json"
    config.save(config_path)
    
    print(f"📋 Enhanced configuration saved to {config_path}")
    print(f"🎯 Training {args.model} with enhanced pipeline")
    print(f"📊 Experiment: {config.experiment.experiment_name}")
    
    # Create enhanced runner
    runner = UnifiedExperimentRunner(config, args.model)
    
    # Load data and start training
    print(f"Loading data from {config.data.data_path}")
    
    if args.model == "property_predictor":
        # Create dataset manager and load data
        dataset_manager = DatasetManager(unified_data_path=config.data.data_path)
        
        # Split dataset
        train_quantum_dataset, val_quantum_dataset, test_quantum_dataset = dataset_manager.split_dataset(
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            test_ratio=config.data.test_split
        )
        
        # Create datasets
        train_dataset = PropertyPredictionDataset(train_quantum_dataset)
        val_dataset = PropertyPredictionDataset(val_quantum_dataset)
        
        print(f"📊 Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print(f"\n🚀 Starting enhanced training with {len(train_loader)} batches per epoch")
        
        # Start training
        runner.train(train_loader, val_loader)
        
    elif args.model == "decision_transformer":
        # Decision Transformer 학습 설정
        from rtg.core.rtg_calculator import create_rtg_calculator
        
        # Property model weights 경로 설정 (best model)
        property_model_path = "weights/best_model.pt"
        
        # RTG 계산기 생성 (property model weights 사용)
        rtg_calculator = create_rtg_calculator(
            checkpoint_path=property_model_path,
            property_weights={
                'entanglement': 1.0,
                'fidelity': 1.0, 
                'expressibility': 1.0
            },
            device=config.model.get_device() if hasattr(config.model, 'get_device') else 'cuda'
        )
        
        # Dataset manager 생성
        dataset_manager = DatasetManager(unified_data_path=config.data.data_path)
        
        # Decision Transformer용 데이터셋 분할
        train_quantum_dataset, val_quantum_dataset, test_quantum_dataset = dataset_manager.split_dataset(
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            test_ratio=config.data.test_split
        )
        
        print(f"📊 Decision Transformer Data: {len(train_quantum_dataset)} train, {len(val_quantum_dataset)} val")
        
        # Decision Transformer 전용 데이터로더 생성
        from training.trainer import QuantumCircuitCollator
        from data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
        
        # 임베딩 파이프라인 설정 (통합 임베딩 시스템 사용)
        embed_config = EmbeddingConfig(
            d_model=config.model.d_model,
            n_gate_types=20,  # Decision Transformer용
            max_seq_len=2000
        )
        embedding_pipeline = create_embedding_pipeline(embed_config)
        
        # 콜레이터 설정 (RTG 계산 포함)
        collator = QuantumCircuitCollator(embedding_pipeline, rtg_calculator)
        
        # 데이터로더 생성
        train_loader = torch.utils.data.DataLoader(
            train_quantum_dataset,
            batch_size=config.training.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_quantum_dataset,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )
        
        print(f"\n🚀 Starting Decision Transformer training with RTG guidance")
        print(f"   Property model weights: {property_model_path}")
        print(f"   Training batches: {len(train_loader)}")
        
        # Decision Transformer 학습 시작
        runner.train_decision_transformer(train_loader, val_loader, rtg_calculator)
    
    print("\n✅ Enhanced training complete!")

if __name__ == "__main__":
    main()
