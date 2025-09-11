"""
Decision Transformer compatible augmented dataset
Applies data augmentation while maintaining episode format compatibility
"""

import torch
import numpy as np
import random
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

from .decision_transformer_dataset import DecisionTransformerDataset
from .base_quantum_dataset import DecisionTransformerCompatDataset
from rtg.core.rtg_calculator import RTGCalculator


class AugmentedDecisionTransformerDataset(DecisionTransformerCompatDataset):
    """Decision Transformer compatible augmented dataset"""
    
    def __init__(self, 
                 base_dataset: DecisionTransformerDataset,
                 noise_samples: int = 500,
                 param_random_samples: int = 1000,
                 noise_std: float = 0.05):
        """
        Initialize augmented Decision Transformer dataset
        
        Args:
            base_dataset: Base DecisionTransformerDataset
            noise_samples: Number of noise augmentation samples
            param_random_samples: Number of parameter randomization samples  
            noise_std: Standard deviation for noise augmentation
        """
        # Initialize parent class first with circuit data from base dataset
        super().__init__(
            circuit_data=base_dataset.circuit_data,
            max_seq_len=base_dataset.max_seq_length if hasattr(base_dataset, 'max_seq_length') else 192,
            rtg_calculator=base_dataset.rtg_calculator if hasattr(base_dataset, 'rtg_calculator') else None,
            enable_rtg=True
        )
        
        self.base_dataset = base_dataset
        self.noise_samples = noise_samples
        self.param_random_samples = param_random_samples
        self.noise_std = noise_std
        
        # Generate augmented episodes and store them for later use
        self.augmented_episodes = self._generate_augmented_episodes()
        print(f"Generated {len(self.augmented_episodes)} augmented episodes")
    def _generate_augmented_episodes(self) -> List[Dict[str, Any]]:
        """Generate augmented episodes maintaining episode format"""
        augmented = []
        base_episodes = self.base_dataset.episodes
        
        print(f"🔄 Generating augmented Decision Transformer episodes...")
        
        # Initialize RTG calculator for recalculating RTG values
        if not hasattr(self, 'rtg_calculator') or self.rtg_calculator is None:
            self.rtg_calculator = RTGCalculator()
            print(f"✅ RTG Calculator initialized for augmented episodes")
        
        # 1. Original episodes
        augmented.extend(base_episodes)
        print(f"📊 Original episodes: {len(base_episodes)}")
        
        # 2. Noise augmented episodes
        noise_episodes = self._generate_noise_episodes(base_episodes, self.noise_samples)
        augmented.extend(noise_episodes)
        print(f"🔊 Noise augmented episodes: {len(noise_episodes)}")
        
        # 3. Parameter randomized episodes  
        param_episodes = self._generate_param_episodes(base_episodes, self.param_random_samples)
        augmented.extend(param_episodes)
        print(f"🎲 Parameter randomized episodes: {len(param_episodes)}")
        
        # Process all augmented episodes to ensure proper formatting and RTG calculation
        for i, episode in enumerate(augmented):
            # Ensure states has proper shape
            if 'states' in episode and len(episode['states'].shape) == 2:
                augmented[i]['states'] = episode['states'].unsqueeze(1)
            
        
        print(f"✅ Total augmented episodes: {len(augmented)}")
        return augmented
    
    def _generate_noise_episodes(self, base_episodes: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Generate noise-augmented episodes"""
        noise_episodes = []
        
        # Check if base_episodes is empty
        if not base_episodes:
            print("⚠️ 기본 에피소드가 없어 노이즈 증강을 건너뜁니다.")
            return []
        
        for i in range(num_samples):
            # Randomly select base episode
            base_episode = random.choice(base_episodes)
            
            # Create noisy copy
            noisy_episode = self._add_noise_to_episode(base_episode, i)
            if noisy_episode:
                noise_episodes.append(noisy_episode)
        
        return noise_episodes
    
    def _generate_param_episodes(self, base_episodes: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Generate parameter-randomized episodes"""
        param_episodes = []
        
        # Check if base_episodes is empty
        if not base_episodes:
            print("⚠️ 기본 에피소드가 없어 파라미터 증강을 건너뜁니다.")
            exit()
        
        for i in range(num_samples):
            # Randomly select base episode
            base_episode = random.choice(base_episodes)
            
            # Create parameter-randomized copy
            param_episode = self._randomize_episode_params(base_episode, i)
            if param_episode:
                param_episodes.append(param_episode)
        
        return param_episodes
    
    def _add_noise_to_episode(self, episode: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Add noise to episode target properties"""
        np.random.seed(seed)
        
        # Deep copy episode
        noisy_episode = {
            'states': episode['states'].clone() if len(episode['states'].shape) == 3 else episode['states'].clone().unsqueeze(1),
            'actions': episode['actions'].clone(),
            'rtg': episode['rtg'].clone(),  # Copy original RTG
            'seq_length': episode['seq_length'],
            'target_properties': episode['target_properties'].copy() if isinstance(episode['target_properties'], dict) else episode['target_properties'].clone(),
            'circuit_id': f"{episode['circuit_id']}_noise_{seed}"
        }
        
        # Add noise to RTG values
        rtg_noise = np.random.normal(0, self.noise_std * 0.1, size=noisy_episode['rtg'].shape)
        noisy_episode['rtg'] = torch.clamp(noisy_episode['rtg'] + torch.tensor(rtg_noise, dtype=torch.float32), 0.0, float('inf'))
        
        # Check if target_properties is a dictionary or tensor
        if isinstance(noisy_episode['target_properties'], dict):
            # Handle dictionary target properties
            target_props = noisy_episode['target_properties']
            
            # Add noise to each property
            if 'entanglement' in target_props:
                entanglement_noise = np.random.normal(0, self.noise_std)
                target_props['entanglement'] += entanglement_noise
                target_props['entanglement'] = max(0.0, min(1.0, target_props['entanglement']))
            
            if 'fidelity' in target_props:
                fidelity_noise = np.random.normal(0, self.noise_std)
                target_props['fidelity'] += fidelity_noise
                target_props['fidelity'] = max(0.0, min(1.0, target_props['fidelity']))
            
            if 'expressibility' in target_props:
                expr_noise = np.random.normal(0, self.noise_std * 5)
                target_props['expressibility'] += expr_noise
                target_props['expressibility'] = max(0.0, target_props['expressibility'])
        else:
            # Original tensor-based handling
            target_props = noisy_episode['target_properties']
            
            # Entanglement noise (0-1 range)
            if len(target_props) > 0:
                entanglement_noise = np.random.normal(0, self.noise_std)
                target_props[0] = torch.clamp(target_props[0] + entanglement_noise, 0.0, 1.0)
            
            # Fidelity noise (0-1 range)  
            if len(target_props) > 1:
                fidelity_noise = np.random.normal(0, self.noise_std)
                target_props[1] = torch.clamp(target_props[1] + fidelity_noise, 0.0, 1.0)
            
            # Expressibility noise (positive values)
            if len(target_props) > 2:
                expr_noise = np.random.normal(0, self.noise_std * 5)
                target_props[2] = torch.clamp(target_props[2] + expr_noise, 0.0, float('inf'))
        
        return noisy_episode
    
    def _randomize_episode_params(self, episode: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Randomize action parameters in episode"""
        np.random.seed(seed)
        
        # Deep copy episode
        param_episode = {
            'states': episode['states'].clone() if len(episode['states'].shape) == 3 else episode['states'].clone().unsqueeze(1),
            'actions': episode['actions'].clone(),
            'rtg': episode['rtg'].clone(),  # Copy original RTG
            'seq_length': episode['seq_length'],
            'target_properties': episode['target_properties'].copy() if isinstance(episode['target_properties'], dict) else episode['target_properties'].clone(),
            'circuit_id': f"{episode['circuit_id']}_param_{seed}"
        }
        
        # Randomize parameters in actions (last dimension)
        actions = param_episode['actions']
        seq_len = episode['seq_length']
        
        # Add small noise to RTG values for parameter randomized episodes
        rtg_noise = np.random.normal(0, self.noise_std * 0.05, size=param_episode['rtg'].shape)
        param_episode['rtg'] = torch.clamp(param_episode['rtg'] + torch.tensor(rtg_noise, dtype=torch.float32), 0.0, float('inf'))
        
        for i in range(seq_len - 1):  # actions has seq_len - 1 elements
            # Handle tensor indexing properly
            action_item = actions[i]
            
            # Check if action is a scalar tensor (0-d) or has multiple dimensions
            if action_item.dim() == 0:
                # Skip scalar actions (no parameters to randomize)
                continue
            elif action_item.dim() == 1 and len(action_item) > 3:
                # Add parameter noise for parameterized gates
                param_noise = np.random.normal(0, 0.1)  # Small parameter variation
                actions[i][3] = torch.clamp(actions[i][3] + param_noise, -np.pi, np.pi)
        
        return param_episode
    
    def __len__(self) -> int:
        return len(self.augmented_episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.augmented_episodes[idx].copy()
        
        # Ensure states has proper 3D shape [seq_len, 1, state_dim]
        if len(episode['states'].shape) == 2:
            episode['states'] = episode['states'].unsqueeze(1)
        
        # Convert target_properties dict to tensor if needed for the collator
        if isinstance(episode['target_properties'], dict):
            props = episode['target_properties']
            episode['target_properties'] = torch.tensor([
                float(props.get('fidelity', 0.5)),
                float(props.get('entanglement', 0.5)),
                float(props.get('expressibility', 0.5))
            ], dtype=torch.float)
        
        return episode
    
    def _get_formatted_item(self, idx: int) -> Dict[str, Any]:
        """Decision Transformer용 에피소드 포맷으로 데이터 반환"""
        # Use the same logic as __getitem__ to ensure consistency
        return self.__getitem__(idx)
    
    def get_target_format(self) -> str:
        return 'decision_transformer'
    
    def get_collate_fn(self):
        """Decision Transformer collator 사용"""
        from .simple_dt_collator import SimpleDecisionTransformerCollator
        return SimpleDecisionTransformerCollator(
            max_seq_length=self.base_dataset.max_seq_length,
            d_model=512  # Default d_model
        )
