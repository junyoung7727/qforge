"""
Streamlined Decision Transformer Dataset
간결한 데이터 흐름: QuantumDataset -> SAR Sequences -> Target Labels
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
import sys
import json
import hashlib
from pathlib import Path

# Add quantumcommon to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "quantumcommon"))

from quantumcommon.gates import gate_registry
from data.quantum_circuit_dataset import CircuitData
from rtg.core.rtg_calculator import RTGCalculator


class StreamlinedDTDataset(Dataset):
    """
    간소화된 Decision Transformer 데이터셋
    
    Data Flow:
    1. QuantumDataset (JSON) -> CircuitData objects
    2. RTG calculation for rewards
    3. Direct SAR sequence generation with GNN embeddings
    4. Target labels for gate prediction
    """
    
    def __init__(
        self,
        circuit_data_list: List[CircuitData],
        rtg_calculator: RTGCalculator,
        target_properties: Dict[str, float],
        max_seq_length: int = 64,
        d_model: int = 256,
        cache_dir: Optional[str] = None
    ):
        self.circuit_data_list = circuit_data_list
        self.rtg_calculator = rtg_calculator
        self.target_properties = target_properties
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
        # Gate vocabulary
        self.gate_vocab = gate_registry.get_gate_vocab()
        self.vocab_size = len(self.gate_vocab)
        
        # Initialize caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/streamlined_dt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rtg_cache = {}
        self.embedding_cache = {}
        
        # Initialize embedding pipeline
        self._init_embedding_pipeline()
        
        # Process all circuits into SAR sequences
        self.sar_episodes = self._process_all_circuits()
        
    def _init_embedding_pipeline(self):
        """Initialize GNN embedding pipeline"""
        from data.embedding_pipeline_refactored import EmbeddingPipeline, EmbeddingConfig
        
        self.embedding_config = EmbeddingConfig(
            d_model=self.d_model,
            n_gate_types=self.vocab_size,
            n_qubits=8,  # Max qubits supported
            max_seq_len=self.max_seq_length
        )
        
        self.embedding_pipeline = EmbeddingPipeline(self.embedding_config)
    
    def _process_all_circuits(self) -> List[Dict[str, torch.Tensor]]:
        """Process all circuits into SAR episodes"""
        episodes = []
        
        for circuit_data in self.circuit_data_list:
            episode = self._create_sar_episode(circuit_data)
            if episode is not None:
                episodes.append(episode)
                
        return episodes
    
    def _create_sar_episode(self, circuit_data: CircuitData) -> Optional[Dict[str, torch.Tensor]]:
        """Create single SAR episode from circuit data"""
        circuit_spec = circuit_data.circuit_spec
        
        if not circuit_spec.gates or len(circuit_spec.gates) == 0:
            return None
            
        circuit_id = circuit_spec.circuit_id
        
        # 1. Calculate RTG values (with caching)
        try:
            rtg_values = self._get_cached_rtg(circuit_spec)
        except Exception as e:
            print(f"RTG calculation failed for {circuit_id}: {e}")
            return None
        
        # 2. Generate embeddings using GNN pipeline (with caching)
        try:
            embeddings = self._get_cached_embeddings(circuit_spec)
        except Exception as e:
            print(f"Embedding generation failed for {circuit_id}: {e}")
            return None
        
        # 3. Create SAR sequence and targets
        sar_sequence, action_targets, attention_mask = self._build_sar_sequence(
            circuit_spec, embeddings, rtg_values
        )
        
        return {
            'circuit_id': circuit_spec.circuit_id,
            'input_sequence': sar_sequence,
            'action_targets': action_targets,
            'attention_mask': attention_mask,
            'rtg_values': torch.tensor(rtg_values, dtype=torch.float32),
            'seq_length': len(circuit_spec.gates),
            'target_properties': self.target_properties
        }
    
    def _build_sar_sequence(
        self, 
        circuit_spec, 
        embeddings: torch.Tensor, 
        rtg_values: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build SAR sequence: [S0, A0, R0, S1, A1, R1, ...]
        
        Returns:
            sar_sequence: [seq_len, d_model] - Input sequence for transformer
            action_targets: [num_actions] - Gate indices for prediction
            attention_mask: [seq_len] - Valid positions mask
        """
        gates = circuit_spec.gates[:self.max_seq_length//3]  # Limit gates to fit SAR pattern
        num_gates = len(gates)
        sar_length = num_gates * 3
        
        # Initialize tensors
        sar_sequence = torch.zeros(self.max_seq_length, self.d_model)
        action_targets = torch.zeros(num_gates, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_length)
        
        # Build SAR pattern
        for i, gate in enumerate(gates):
            base_idx = i * 3
            
            # State (S): Current circuit state
            if base_idx < embeddings.shape[0]:
                sar_sequence[base_idx] = embeddings[base_idx]
            attention_mask[base_idx] = 1
            
            # Action (A): Gate to be predicted
            gate_idx = self._get_gate_index(gate)
            action_targets[i] = gate_idx
            
            if base_idx + 1 < self.max_seq_length:
                # Use gate embedding or learned action embedding
                if base_idx + 1 < embeddings.shape[0]:
                    sar_sequence[base_idx + 1] = embeddings[base_idx + 1]
                attention_mask[base_idx + 1] = 1
            
            # Reward (R): RTG value embedding
            if base_idx + 2 < self.max_seq_length and i < len(rtg_values):
                rtg_embedding = self._encode_rtg_value(rtg_values[i])
                sar_sequence[base_idx + 2] = rtg_embedding
                attention_mask[base_idx + 2] = 1
        
        return sar_sequence, action_targets, attention_mask
    
    def _get_gate_index(self, gate) -> int:
        """Get valid gate index from vocabulary"""
        gate_name = gate.name.lower()
        gate_idx = self.gate_vocab.get(gate_name, 0)  # Default to first gate if unknown
        return max(0, min(gate_idx, self.vocab_size - 1))  # Clamp to valid range
    
    def _encode_rtg_value(self, rtg_value: float) -> torch.Tensor:
        """Encode RTG value as embedding vector"""
        # Simple encoding: broadcast RTG value across embedding dimension
        rtg_embedding = torch.full((self.d_model,), rtg_value, dtype=torch.float32)
        return rtg_embedding
    
    def __len__(self) -> int:
        return len(self.sar_episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sar_episodes[idx]
    
    def get_vocab_size(self) -> int:
        """Return gate vocabulary size for model configuration"""
        return self.vocab_size
    
    def get_collate_fn(self):
        """Return collate function for DataLoader"""
        def collate_fn(batch):
            """Simple batch collation - tensors are already properly shaped"""
            if not batch:
                return {}
                
            # Stack tensors
            collated = {}
            for key in batch[0].keys():
                if key in ['circuit_id', 'target_properties']:
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = torch.stack([item[key] for item in batch])
            
            return collated
        
        return collate_fn
    
    def _get_circuit_hash(self, circuit_spec) -> str:
        """Generate hash for circuit caching"""
        circuit_str = f"{circuit_spec.circuit_id}_{len(circuit_spec.gates)}_{circuit_spec.num_qubits}"
        for gate in circuit_spec.gates:
            gate_str = f"{gate.name}_{getattr(gate, 'qubits', [])}_{getattr(gate, 'parameters', [])}"
            circuit_str += gate_str
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    def _get_cached_rtg(self, circuit_spec) -> List[float]:
        """Get RTG values with caching"""
        circuit_hash = self._get_circuit_hash(circuit_spec)
        
        # Check memory cache
        if circuit_hash in self.rtg_cache:
            return self.rtg_cache[circuit_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"rtg_{circuit_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    rtg_values = cached_data['rtg_values']
                    self.rtg_cache[circuit_hash] = rtg_values
                    return rtg_values
            except Exception as e:
                print(f"Failed to load RTG cache: {e}")
        
        # Calculate RTG values
        rtg_values, rewards, properties = self.rtg_calculator.calculate_rtg_sequence(
            circuit_spec, self.target_properties
        )
        
        # Cache results
        self.rtg_cache[circuit_hash] = rtg_values
        
        # Save to disk
        try:
            cache_data = {
                'circuit_id': circuit_spec.circuit_id,
                'rtg_values': rtg_values,
                'rewards': rewards,
                'properties': properties,
                'target_properties': self.target_properties
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save RTG cache: {e}")
        
        return rtg_values
    
    def _get_cached_embeddings(self, circuit_spec) -> torch.Tensor:
        """Get embeddings with caching"""
        circuit_hash = self._get_circuit_hash(circuit_spec)
        
        # Check memory cache
        if circuit_hash in self.embedding_cache:
            return self.embedding_cache[circuit_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"emb_{circuit_hash}.pt"
        if cache_file.exists():
            try:
                embeddings = torch.load(cache_file)
                self.embedding_cache[circuit_hash] = embeddings
                return embeddings
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
        
        # Generate embeddings
        result = self.embedding_pipeline.process_circuit(circuit_spec)
        embeddings = result['decision_transformer']['input_sequence'][0]  # Remove batch dim
        
        # Cache results
        self.embedding_cache[circuit_hash] = embeddings
        
        # Save to disk
        try:
            torch.save(embeddings, cache_file)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")
        
        return embeddings
    
    def clear_cache(self):
        """Clear all caches"""
        self.rtg_cache.clear()
        self.embedding_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
        
        print(f"✅ Cache cleared: {self.cache_dir}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        rtg_disk_files = len(list(self.cache_dir.glob("rtg_*.json")))
        emb_disk_files = len(list(self.cache_dir.glob("emb_*.pt")))
        
        return {
            'rtg_memory_cache': len(self.rtg_cache),
            'embedding_memory_cache': len(self.embedding_cache),
            'rtg_disk_cache': rtg_disk_files,
            'embedding_disk_cache': emb_disk_files
        }
