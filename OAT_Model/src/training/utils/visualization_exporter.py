"""
Visualization Data Export Module

학습 메트릭 시각화를 위한 데이터 내보내기 모듈
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class VisualizationExporter:
    """시각화 데이터 내보내기 관리자"""
    
    def __init__(self, save_dir: Path, device: str):
        self.save_dir = save_dir
        self.device = device
    
    def export_visualization_data(self, training_history: List[Dict], config: Any) -> bool:
        """시각화를 위한 메트릭 데이터 저장"""
        try:
            # 시각화용 데이터 구조 생성
            visualization_data = {
                'metadata': {
                    'experiment_name': 'property_prediction_training',
                    'timestamp': datetime.now().isoformat(),
                    'total_epochs': len(training_history),
                    'device': str(self.device),
                    'model_config': {
                        'd_model': getattr(config, 'd_model', 512),
                        'n_heads': getattr(config, 'n_heads', 8),
                        'n_layers': getattr(config, 'n_layers', 6),
                        'attention_mode': getattr(config, 'attention_mode', 'advanced')
                    }
                },
                'metrics': {
                    'epochs': [],
                    'train_loss': [],
                    'val_loss': [],
                    'learning_rate': [],
                    'duration_sec': [],
                    'properties': {
                        'entanglement': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        },
                        'fidelity': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        },
                        'expressibility': {
                            'train_loss': [], 'val_loss': [],
                            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_corr': []
                        }
                    }
                }
            }
            
            # 에포크별 데이터 추출
            for epoch_data in training_history:
                visualization_data['metrics']['epochs'].append(epoch_data.get('epoch', 0))
                visualization_data['metrics']['train_loss'].append(epoch_data.get('train_loss', 0.0))
                visualization_data['metrics']['val_loss'].append(epoch_data.get('val_loss', 0.0))
                visualization_data['metrics']['learning_rate'].append(epoch_data.get('learning_rate', 0.0))
                visualization_data['metrics']['duration_sec'].append(epoch_data.get('duration_sec', 0.0))
                
                # 프로퍼티별 메트릭 추출
                for prop in ['entanglement', 'fidelity', 'expressibility']:
                    prop_data = visualization_data['metrics']['properties'][prop]
                    prop_data['train_loss'].append(epoch_data.get(f'train_{prop}', 0.0))
                    prop_data['val_loss'].append(epoch_data.get(f'val_{prop}', 0.0))
                    
                    # 정확도 메트릭들
                    for metric in ['mae', 'rmse', 'r2', 'corr']:
                        key = f'val_{prop}_{metric}'
                        prop_data[f'val_{metric}'].append(epoch_data.get(key, 0.0))
            
            # 시각화 데이터 저장
            viz_file = self.save_dir / 'visualization_data.json'
            with open(viz_file, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 시각화 데이터 저장 완료: {viz_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 시각화 데이터 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_training_summary(self, training_history: List[Dict], best_val_loss: float, 
                              early_stopped: bool = False) -> bool:
        """학습 요약 통계 저장"""
        try:
            if not training_history:
                return False
            
            summary = {
                'training_overview': {
                    'total_epochs': len(training_history),
                    'best_epoch': 0,
                    'best_val_loss': best_val_loss,
                    'early_stopped': early_stopped,
                    'final_learning_rate': training_history[-1].get('learning_rate', 0.0)
                },
                'loss_progression': {
                    'initial_train_loss': training_history[0].get('train_loss', 0.0),
                    'final_train_loss': training_history[-1].get('train_loss', 0.0),
                    'initial_val_loss': training_history[0].get('val_loss', 0.0),
                    'final_val_loss': training_history[-1].get('val_loss', 0.0)
                },
                'property_performance': {}
            }
            
            # 최적 에포크 찾기
            best_val_loss_found = float('inf')
            for i, epoch_data in enumerate(training_history):
                if epoch_data.get('val_loss', float('inf')) < best_val_loss_found:
                    best_val_loss_found = epoch_data.get('val_loss', float('inf'))
                    summary['training_overview']['best_epoch'] = i
            
            # 프로퍼티별 최종 성능
            final_epoch = training_history[-1]
            for prop in ['entanglement', 'fidelity', 'expressibility']:
                prop_summary = {}
                for metric in ['mae', 'rmse', 'r2', 'corr']:
                    key = f'val_{prop}_{metric}'
                    if key in final_epoch:
                        prop_summary[f'final_{metric}'] = final_epoch[key]
                
                if prop_summary:
                    summary['property_performance'][prop] = prop_summary
            
            # 요약 통계 저장
            summary_file = self.save_dir / 'training_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📊 학습 요약 저장 완료: {summary_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 학습 요약 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
