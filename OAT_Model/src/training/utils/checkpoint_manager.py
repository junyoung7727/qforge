"""
Checkpoint Management Module

모델 체크포인트 저장 및 로딩을 담당하는 모듈
"""
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict


class CheckpointManager:
    """체크포인트 저장 및 로딩 관리자"""
    
    def __init__(self, save_dir: str, device: torch.device):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       config: Any,
                       training_state: Dict[str, Any],
                       filename: str) -> bool:
        """체크포인트 저장 - 메타데이터 추가 및 보안 개선"""
        try:
            # 현재 GPU 메모리 사용량 확인
            gpu_memory_info = None
            if self.device.type == 'cuda':
                gpu_memory_info = {
                    'allocated': torch.cuda.memory_allocated() / (1024**3),
                    'reserved': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
                }
            
            # 체크포인트 데이터 구성
            checkpoint = {
                # 학습 상태
                'epoch': training_state.get('current_epoch', 0),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': training_state.get('best_val_loss', float('inf')),
                
                # 구성 및 기록
                'config': asdict(config) if hasattr(config, '__dict__') else config,
                'training_history': training_state.get('training_history', []),
                
                # 추가 메타데이터
                'early_stopping': {
                    'counter': training_state.get('early_stopping_counter', 0),
                    'patience': training_state.get('patience', 15),
                    'min_delta': training_state.get('min_delta', 0.001),
                    'stopped_early': training_state.get('early_stopped', False)
                },
                
                # 시스템 정보
                'timestamp': time.time(),
                'save_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'gpu_memory': gpu_memory_info
            }
            
            # 체크포인트 저장 경로 구성
            checkpoint_path = self.save_dir / filename
            
            # 임시 파일로 저장 후 이동 (파일 손상 방지)
            temp_path = self.save_dir / f"temp_{filename}"
            torch.save(checkpoint, temp_path)
            
            # 이미 파일이 있는 경우 백업
            if checkpoint_path.exists():
                backup_path = self.save_dir / f"backup_{filename}"
                if backup_path.exists():
                    backup_path.unlink()  # 기존 백업 삭제
                checkpoint_path.rename(backup_path)  # 기존 파일을 백업으로 이동
            
            # 임시 파일을 최종 파일로 이름 변경
            temp_path.rename(checkpoint_path)
            
            # 성공 메시지
            if 'best_model' in filename:
                print(f"[DONE] 체크포인트 저장 완료: {checkpoint_path.name}")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] 체크포인트 저장 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        체크포인트를 불러와 모델과 학습 상태를 복원
        
        Args:
            model: 모델 인스턴스
            optimizer: 옵티마이저 인스턴스
            scheduler: 스케줄러 인스턴스
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            Dict[str, Any]: 복원된 학습 상태 정보, 실패시 None
        """
        try:
            # Path 객체로 변환
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f"\n[ERROR] 체크포인트 파일이 없습니다: {checkpoint_path}")
                return None
                
            # CPU에서 로딩 (안정성을 위해)
            print(f"\n[LOAD] 체크포인트 로딩 중: {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 기본 필수 필드 검색
            required_fields = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for field in required_fields:
                if field not in checkpoint:
                    print(f"\n[ERROR] 체크포인트에 필수 필드 '{field}'가 없습니다.")
                    return None
            
            # 상세 정보 출력 (메타데이터)
            if 'save_date' in checkpoint:
                print(f"  - 저장 시점: {checkpoint['save_date']}")
            print(f"  - 에폭: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  - 최적 검증 손실: {checkpoint['best_val_loss']:.6f}")
            
            # GPU 메모리 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 모델 가중치 로딩
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 추가 상태 복원
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 스케줄러 복원
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"  ✅ 스케줄러 복원 완료")
                except Exception as e:
                    print(f"  ❌ 스케줄러 복원 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"스케줄러 복원 중 치명적 오류: {e}") from e
            
            # 모델을 적절한 기기로 이동
            model.to(self.device)
            
            # 학습 상태 정보 반환
            training_state = {
                'current_epoch': checkpoint['epoch'],
                'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                'training_history': checkpoint.get('training_history', []),
                'early_stopping_counter': checkpoint.get('early_stopping', {}).get('counter', 0),
                'patience': checkpoint.get('early_stopping', {}).get('patience', 15),
                'min_delta': checkpoint.get('early_stopping', {}).get('min_delta', 0.001),
                'early_stopped': checkpoint.get('early_stopping', {}).get('stopped_early', False)
            }
            
            print(f"\n✅ 체크포인트 로딩 성공! 학습을 에폭 {training_state['current_epoch']+1}부터 계속합니다.")
            return training_state
            
        except Exception as e:
            print(f"\n⚠️ 체크포인트 로딩 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_training_history(self, training_history: list, filename: str = 'training_history.json'):
        """학습 기록 저장"""
        import json
        
        history_path = self.save_dir / filename
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"[SAVE] 학습 기록 저장: {history_path}")
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """체크포인트 파일 경로 반환"""
        return self.save_dir / filename
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """오래된 체크포인트 파일 정리"""
        try:
            # checkpoint_epoch_*.pt 패턴의 파일들 찾기
            checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
            
            if len(checkpoint_files) > keep_count:
                # 파일명에서 에폭 번호 추출하여 정렬
                checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                
                # 오래된 파일들 삭제
                files_to_delete = checkpoint_files[:-keep_count]
                for file_path in files_to_delete:
                    file_path.unlink()
                    print(f"[CLEANUP] 오래된 체크포인트 삭제: {file_path.name}")
                
        except Exception as e:
            print(f"[WARNING] 체크포인트 정리 중 오류: {e}")
