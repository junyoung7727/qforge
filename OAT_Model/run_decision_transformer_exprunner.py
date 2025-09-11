#!/usr/bin/env python3
"""
Decision Transformer Training with Unified ExpRunner
프로퍼티 모델의 best weights를 RTG 가이드로 사용하는 디시전 트랜스포머 학습
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def main():
    parser = argparse.ArgumentParser(description="Decision Transformer Training with ExpRunner")
    
    # 기본 설정
    parser.add_argument("--config", type=str, default="medium", 
                       choices=["small", "medium", "large"],
                       help="Model size configuration")
    parser.add_argument("--data-path", type=str, 
                       default="raw_data/merged_data.json",
                       help="Path to training data")
    parser.add_argument("--embedding-mode", type=str, default="gnn",
                       choices=["gnn", "grid", "hybrid"],
                       help="Embedding mode")
    parser.add_argument("--attention-mode", type=str, default="advanced",
                       choices=["standard", "advanced", "hybrid"],
                       help="Attention mode")
    
    # Decision Transformer 특화 설정
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size (reduced for DT)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--property-weights-path", type=str,
                       default="weights/best_model.pt",
                       help="Path to property model weights for RTG guidance")
    
    # 실험 설정
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # unified_experiment_runner 실행 (use hyphenated args for unified_experiment_runner)
    cmd_args = [
        "python3", "src/unified_experiment_runner.py",
        "--model", "decision_transformer",
        "--config", args.config,
        "--data-path", args.data_path,
        "--embedding-mode", args.embedding_mode,
        "--attention-mode", args.attention_mode
    ]
    
    if args.experiment_name:
        cmd_args.extend(["--experiment-name", args.experiment_name])
    
    print("🚀 Starting Decision Transformer Training with ExpRunner")
    print(f"   Model Size: {args.config}")  
    print(f"   Data Path: {args.data_path}")
    print(f"   Property Weights: {args.property_weights_path}")
    print(f"   Embedding Mode: {args.embedding_mode}")
    print(f"   Attention Mode: {args.attention_mode}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print()
    
    # 명령어 실행
    import subprocess
    result = subprocess.run(cmd_args, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ Decision Transformer training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
