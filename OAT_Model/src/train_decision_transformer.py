"""
Decision Transformer 훈련 실행 스크립트
인자를 받아 훈련을 실행하는 메인 스크립트
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.decision_transformer import create_decision_transformer
from training.decision_transformer_trainer import DecisionTransformerTrainer
from rtg.core.rtg_calculator import RTGCalculator, create_rtg_calculator
from data.simple_dt_collator import SimpleDecisionTransformerCollator
from data.streamlined_dt_dataset import StreamlinedDTDataset
from data.quantum_circuit_dataset import DatasetManager, DecisionTransformerDataset
from config.model_configs import get_model_config, print_model_info, get_available_sizes


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Decision Transformer Training")
    
    # 기본 설정
    parser.add_argument('--data_path', type=str, default=r'C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json', help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    
    # 모델 사이즈 (표준화)
    parser.add_argument('--model_size', type=str, default='M', choices=['S', 'M', 'L'], 
                       help='Model size: S(Small), M(Medium), L(Large)')
    
    # 어텐션 모드
    parser.add_argument('--attention_mode', type=str, default='advanced', 
                       choices=['standard', 'advanced'], 
                       help='Attention mode: standard or advanced')
    
    # 프로퍼티 모델 가중치
    parser.add_argument('--property_model_path', type=str, 
                       help='Path to pre-trained property prediction model weights')
    
    # 임베딩 모드
    parser.add_argument('--embedding_mode', type=str, default='gnn', 
                       choices=['gnn', 'transformer', 'hybrid', 'simple'], 
                       help='Embedding mode: gnn, transformer, hybrid, or simple')
    
    # 데이터 증강 옵션
    parser.add_argument('--use_augmentation', default=True,action='store_true', 
                       help='Enable data augmentation for training dataset')
    parser.add_argument('--noise_samples', type=int, default=500,
                       help='Number of noise augmentation samples per circuit')
    parser.add_argument('--param_random_samples', type=int, default=1000,
                       help='Number of parameter randomization samples per circuit')
    
    # 훈련 파라미터
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # 디바이스 및 기타
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_embedding_config(embedding_mode, base_config):
    """임베딩 모드별 설정 생성"""
    embedding_config = {}
    
    if embedding_mode == 'gnn':
        embedding_config = {
            'type': 'gnn',
            'hidden_dim': base_config['d_model'],
            'num_layers': 3,
            'dropout': 0.1,
            'aggregation': 'mean',
            'use_edge_features': True
        }
    elif embedding_mode == 'transformer':
        embedding_config = {
            'type': 'transformer',
            'hidden_dim': base_config['d_model'],
            'num_heads': base_config['n_heads'],
            'num_layers': 2,
            'dropout': 0.1
        }
    elif embedding_mode == 'hybrid':
        embedding_config = {
            'type': 'hybrid',
            'gnn_layers': 2,
            'transformer_layers': 1,
            'hidden_dim': base_config['d_model'],
            'num_heads': base_config['n_heads'],
            'dropout': 0.1
        }
    elif embedding_mode == 'simple':
        embedding_config = {
            'type': 'simple',
            'hidden_dim': base_config['d_model'],
            'num_layers': 2,
            'dropout': 0.1
        }
    else:
        raise ValueError(f"Unknown embedding mode: {embedding_mode}")
    
    return embedding_config


def create_model(model_config, property_model_path=None, embedding_mode='gnn'):
    """모델 생성"""
    print("🤖 Decision Transformer 모델 생성 중...")
    
    # Property prediction model 로드 (필요한 경우)
    property_prediction_model = None
    if property_model_path:
        try:
            from rtg.model_loader import load_property_predictor
            property_prediction_model = load_property_predictor(property_model_path)
            print(f"✅ Property prediction model 로드 완료: {property_model_path}")
        except Exception as e:
            print(f"⚠️ Property prediction model 로드 실패: {e}")
            property_prediction_model = None
    
    # SAR 시퀀스는 3배 길이
    dt_config = model_config.copy()
    dt_config['max_seq_length'] = model_config['max_seq_length'] * 3
    
    # 임베딩 모드 처리
    embedding_config = get_embedding_config(embedding_mode, dt_config)
    
    # Decision Transformer 생성
    dt_config['property_prediction_model'] = property_prediction_model
    dt_config['embedding_mode'] = embedding_mode
    dt_config['embedding_config'] = embedding_config
    model = create_decision_transformer(dt_config)
    
    print(f"   - 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 모델과 함께 property_prediction_model도 반환하여 재사용
    return model, property_prediction_model


def create_real_data_loaders(args, model_config, property_prediction_model=None):
    """실제 데이터 로더 생성"""
    print("📊 실제 데이터 로더 생성 중...")
    
    # 데이터 파일 확인
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")
    
    # Quantum dataset 로드 (한 번만)
    from data.quantum_circuit_dataset import DatasetManager
    manager = DatasetManager(unified_data_path=args.data_path)
    circuit_data = manager.merge_data()
    quantum_dataset = manager
    
    # RTG 계산기 생성 - 이미 로드된 property_prediction_model 재사용
    rtg_calculator = create_rtg_calculator(
        checkpoint_path=args.property_model_path,
        device=args.device,
        loaded_model=property_prediction_model
    )
    
    # 간소화된 데이터셋 생성 - 직접 circuit_data 사용
    target_properties = {'entanglement': 0.8, 'expressibility': 0.7}
    base_dataset = StreamlinedDTDataset(
        circuit_data_list=circuit_data,
        rtg_calculator=rtg_calculator,
        target_properties=target_properties,
        max_seq_length=model_config['max_seq_length'],
        d_model=model_config.get('d_model', 512)
    )
    
    # 훈련/검증 분할
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        base_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 데이터 증강 적용 (훈련 데이터만)
    if args.use_augmentation:
        print(f"🔄 데이터 증강 적용 중...")
        print(f"   - 노이즈 샘플: {args.noise_samples}")
        print(f"   - 파라미터 랜덤 샘플: {args.param_random_samples}")
        
        # Create augmented dataset from base dataset before splitting
        augmented_base_dataset = AugmentedDecisionTransformerDataset(
            base_dataset,
            noise_samples=args.noise_samples,
            param_random_samples=args.param_random_samples
        )
        
        # Re-split the augmented dataset
        aug_train_size = int(0.8 * len(augmented_base_dataset))
        aug_val_size = len(augmented_base_dataset) - aug_train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            augmented_base_dataset, [aug_train_size, aug_val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"   - 증강된 훈련 샘플: {len(train_dataset):,}")
        print(f"   - 증강된 검증 샘플: {len(val_dataset):,}")
    else:
        print("   - 데이터 증강 비활성화")
    
    # 간소화된 콜레이터 사용
    collate_fn = base_dataset.get_collate_fn()
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False
    )
    
    print(f"   - 훈련 샘플: {len(train_dataset):,}")
    print(f"   - 검증 샘플: {len(val_dataset):,}")
    print(f"   - 배치 크기: {args.batch_size}")
    
    return train_loader, val_loader


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Decision Transformer 훈련 시작")
    print(f"   - 모델 사이즈: {args.model_size}")
    print(f"   - 어텐션 모드: {args.attention_mode}")
    print(f"   - 임베딩 모드: {args.embedding_mode}")
    print(f"   - 데이터 경로: {args.data_path}")
    print(f"   - 출력 디렉토리: {args.output_dir}")
    print(f"   - 디바이스: {args.device}")
    print(f"   - 에포크: {args.epochs}")
    print(f"   - 배치 크기: {args.batch_size}")
    if args.property_model_path:
        print(f"   - 프로퍼티 모델: {args.property_model_path}")
    
    # 임베딩 모드별 세부 설정 출력
    embedding_details = {
        'gnn': 'Graph Neural Network 기반 회로 임베딩',
        'transformer': 'Transformer 기반 시퀀셜 임베딩', 
        'hybrid': 'GNN + Transformer 하이브리드 임베딩',
        'simple': '단순 선형 임베딩'
    }
    print(f"📊 임베딩 모드: {args.embedding_mode} - {embedding_details[args.embedding_mode]}")
    print()
    
    # 모델 설정 로드
    model_config = get_model_config(args.model_size)
    model_config['attention_mode'] = args.attention_mode
    print_model_info(args.model_size)
    print(f"   - Attention Mode: {args.attention_mode}")
    print(f"   - Embedding Mode: {args.embedding_mode}")
    if args.property_model_path:
        print(f"   - Property Model: {args.property_model_path}")
    
    # 훈련 설정 업데이트
    training_config = {
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.epochs,
        'min_lr': 1e-6,
        'batch_size': args.batch_size
    }
    
    # 모델 생성 (property_prediction_model 먼저 로드)
    model, property_prediction_model = create_model(model_config, args.property_model_path, args.embedding_mode)
    
    # 실제 데이터 로더 생성 (이미 로드된 property_prediction_model 재사용)
    train_loader, val_loader = create_real_data_loaders(args, model_config, property_prediction_model)
    
    # 트레이너 생성
    trainer = DecisionTransformerTrainer(
        model=model,
        config=training_config,
        device=args.device,
        use_wandb=args.wandb
    )
    
    # 체크포인트에서 재시작
    if args.resume:
        print(f"📂 체크포인트 로드: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 훈련 실행
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=args.output_dir
        )
        
        print("🎉 훈련 완료!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 훈련이 중단되었습니다.")
        # 현재 상태 저장
        checkpoint_path = output_dir / "interrupted_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"💾 중단된 상태가 저장되었습니다: {checkpoint_path}")
    
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
