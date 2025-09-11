"""
Embedding Pipeline Module - Backward Compatibility Wrapper
Uses modular components for cleaner architecture
"""

# Import the refactored pipeline
from .embedding_pipeline_refactored import EmbeddingPipeline as RefactoredEmbeddingPipeline
from .embedding_pipeline_refactored import EmbeddingConfig

# Re-export for backward compatibility
EmbeddingPipeline = RefactoredEmbeddingPipeline

# Legacy imports for compatibility
import torch
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import dependencies with fallback paths
try:
    from .quantum_circuit_dataset import CircuitSpec
    from ..encoding.grid_graph_encoder import GridGraphEncoder
except ImportError:
    try:
        from .quantum_circuit_dataset import CircuitSpec
        from ..encoding.grid_graph_encoder import GridGraphEncoder
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from data.quantum_circuit_dataset import CircuitSpec
        from encoding.grid_graph_encoder import GridGraphEncoder

# Gate registry import
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry

# For any legacy code that might import specific classes
from .cache_manager import BatchMetadata

# Legacy compatibility - all functionality now provided by RefactoredEmbeddingPipeline
# Original 763-line implementation has been modularized into:
# - cache_manager.py: CacheManager class
# - circuit_processor.py: CircuitProcessor class  
# - batch_processor.py: BatchProcessor class
# - embedding_pipeline_refactored.py: Clean main pipeline class

def create_embedding_pipeline(config: EmbeddingConfig = None, enable_cache: bool = True) -> EmbeddingPipeline:
    """임베딩 파이프라인 팩토리 함수 (캐싱 지원)"""
    if config is None:
        config = EmbeddingConfig()
    
    # 캐싱이 활성화된 임베딩 파이프라인 생성
    pipeline = EmbeddingPipeline(config, enable_cache=enable_cache)
    
    if enable_cache:
        print(f"임베딩 파이프라인 생성 완료! (캐싱: {enable_cache})")
    
    return pipeline
