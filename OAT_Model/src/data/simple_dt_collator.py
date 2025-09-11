"""
Simple Decision Transformer Batch Collator
간단하고 효율적인 배치 처리
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
torch.set_printoptions(threshold=torch.inf, linewidth=torch.inf)


class SimpleDecisionTransformerCollator:
    """Decision Transformer용 간단한 배치 콜레이터"""
    
    def __init__(self, max_seq_length: int = 64, d_model: int = 256):
        self.max_seq_length = max_seq_length
        self.d_model = d_model
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 처리"""
        batch_size = len(batch)
        
        # 각 배치 요소에서 데이터 추출
        states_list = []
        actions_list = []
        rtg_list = []
        seq_lengths = []
        target_props_list = []
        
        for item in batch:
            # Ensure consistent tensor shapes
            states = item['states']
            actions = item['actions']
            rtg = item['rtg']
            
            # Normalize states tensor shape to [seq_len, 1, state_dim]
            if len(states.shape) == 1:
                # Shape [seq_len] -> [seq_len, 1, 1]
                states = states.unsqueeze(-1).unsqueeze(-1)
            elif len(states.shape) == 2:
                # Shape [seq_len, state_dim] -> [seq_len, 1, state_dim]
                states = states.unsqueeze(1)
            elif len(states.shape) == 3:
                # Already correct shape [seq_len, 1, state_dim]
                pass
            else:
                raise ValueError(f"Unexpected states tensor shape: {states.shape}")
            
            # Normalize actions tensor shape to [seq_len-1, action_dim]
            if len(actions.shape) == 1:
                # Shape [seq_len-1] -> [seq_len-1, 1]
                actions = actions.unsqueeze(-1)
            
            # Normalize rtg tensor shape to [seq_len]
            if len(rtg.shape) == 0:
                # Scalar -> [1]
                rtg = rtg.unsqueeze(0)
            
            states_list.append(states)
            actions_list.append(actions)
            rtg_list.append(rtg)
            seq_lengths.append(item['seq_length'])
            target_props_list.append(item['target_properties'])
        
        # Find maximum dimensions for padding
        max_seq_len = max(s.shape[0] for s in states_list) if states_list else 1
        max_state_dim = max(s.shape[2] for s in states_list) if states_list else 1
        max_action_dim = max(a.shape[1] if len(a.shape) > 1 else 1 for a in actions_list) if actions_list else 1
        
        # Ensure minimum dimensions
        max_seq_len = max(max_seq_len, 1)
        max_state_dim = max(max_state_dim, 1)
        max_action_dim = max(max_action_dim, 1)
        
        # Pad tensors to consistent shapes
        padded_states = []
        padded_actions = []
        padded_rtg = []
        
        for i in range(batch_size):
            states = states_list[i]
            actions = actions_list[i]
            rtg = rtg_list[i]
            
            # Pad states to [max_seq_len, 1, max_state_dim]
            seq_len, _, state_dim = states.shape
            if seq_len == 0:  # Handle empty states
                padded_state = torch.zeros(max_seq_len, 1, max_state_dim, dtype=states.dtype)
                padded_states.append(padded_state)
            elif seq_len < max_seq_len or state_dim < max_state_dim:
                padded_state = torch.zeros(max_seq_len, 1, max_state_dim, dtype=states.dtype)
                padded_state[:seq_len, :, :state_dim] = states
                padded_states.append(padded_state)
            else:
                padded_states.append(states[:max_seq_len, :, :max_state_dim])
            
            # Pad actions to [max_seq_len, max_action_dim] - same as states
            action_seq_len = actions.shape[0]
            action_dim = actions.shape[1] if len(actions.shape) > 1 else 1
            target_action_len = max_seq_len  # Same length as states
            
            if action_seq_len == 0:  # Handle empty actions
                padded_action = torch.zeros(target_action_len, max_action_dim, dtype=actions.dtype)
                padded_actions.append(padded_action)
            elif action_seq_len < target_action_len or action_dim < max_action_dim:
                padded_action = torch.zeros(target_action_len, max_action_dim, dtype=actions.dtype)
                if len(actions.shape) > 1:
                    copy_len = min(action_seq_len, target_action_len)
                    padded_action[:copy_len, :action_dim] = actions[:copy_len]
                else:
                    copy_len = min(action_seq_len, target_action_len)
                    padded_action[:copy_len, 0] = actions[:copy_len]
                padded_actions.append(padded_action)
            else:
                if len(actions.shape) == 1:
                    actions = actions.unsqueeze(-1)
                padded_actions.append(actions[:target_action_len, :max_action_dim])
            
            # Pad rtg to [max_seq_len]
            rtg_len = rtg.shape[0]
            if rtg_len == 0:  # Handle empty RTG
                padded_rtg_tensor = torch.zeros(max_seq_len, dtype=rtg.dtype)
                padded_rtg.append(padded_rtg_tensor)
            elif rtg_len < max_seq_len:
                padded_rtg_tensor = torch.zeros(max_seq_len, dtype=rtg.dtype)
                padded_rtg_tensor[:rtg_len] = rtg
                padded_rtg.append(padded_rtg_tensor)
            else:
                padded_rtg.append(rtg[:max_seq_len])
        
        # Stack padded tensors
        states = torch.stack(padded_states)  # [batch_size, max_seq_len, 1, max_state_dim]
        actions = torch.stack(padded_actions)  # [batch_size, max_seq_len, max_action_dim]
        rtg = torch.stack(padded_rtg)  # [batch_size, max_seq_len]
        target_properties = torch.stack(target_props_list)  # [batch_size, 3]
        
        # 원본 액션 데이터 보존 (손실 계산용) - 배치에서 실제 데이터 추출
        original_actions = actions.clone()
        
        # positions와 parameters 데이터 추출
        positions_list = []
        parameters_list = []
        
        for item in batch:
            if 'positions' in item and item['positions'] is not None:
                positions_list.append(item['positions'])
            else:
                # 기본값: actions에서 위치 정보 추출 (첫 2개 요소)
                item_actions = item['actions']
                if len(item_actions.shape) > 1 and item_actions.shape[1] >= 2:
                    positions_list.append(item_actions[:, :2])
                else:
                    positions_list.append(torch.zeros(len(item_actions), 2, dtype=torch.long))
            
            if 'parameters' in item and item['parameters'] is not None:
                parameters_list.append(item['parameters'])
            else:
                # 기본값: 0으로 채움
                parameters_list.append(torch.zeros(len(item['actions']), dtype=torch.float32))
        
        # 패딩 후 스택
        max_pos_len = max(len(pos) for pos in positions_list)
        max_param_len = max(len(param) for param in parameters_list)
        
        padded_positions = []
        padded_parameters = []
        
        for i in range(batch_size):
            # positions 패딩
            pos = positions_list[i]
            if len(pos) < max_pos_len:
                padded_pos = torch.zeros(max_pos_len, 2, dtype=torch.long)
                padded_pos[:len(pos)] = pos
                padded_positions.append(padded_pos)
            else:
                padded_positions.append(pos[:max_pos_len])
            
            # parameters 패딩
            param = parameters_list[i]
            if len(param) < max_param_len:
                padded_param = torch.zeros(max_param_len, dtype=torch.float32)
                padded_param[:len(param)] = param
                padded_parameters.append(padded_param)
            else:
                padded_parameters.append(param[:max_param_len])
        
        original_positions = torch.stack(padded_positions)
        original_parameters = torch.stack(padded_parameters)
        
        # 어텐션 마스크 생성
        attention_mask = torch.zeros(batch_size, self.max_seq_length, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            attention_mask[i, :seq_len] = True
        
        # SAR 시퀀스 구성: State-Action-RTG 인터리빙
        batch_size, max_seq_len, _, state_dim = states.shape  # states is now [batch_size, max_seq_len, 1, max_state_dim]
        states = states.squeeze(2)  # Remove the middle dimension: [batch_size, max_seq_len, max_state_dim]
        action_dim = actions.shape[-1] if len(actions.shape) > 2 else 1
        
        # 콜레이터 초기화 시 설정된 d_model 사용
        d_model = self.d_model
        
        # SAR 시퀀스 텐서 초기화 - 동적 길이로 조정
        max_sar_len = max(seq_lengths) * 3  # 실제 최대 시퀀스 길이의 3배
        input_sequence = torch.zeros(batch_size, max_sar_len, d_model)
        action_prediction_mask = torch.zeros(batch_size, max_sar_len, dtype=torch.bool)
        
        for b in range(batch_size):
            actual_seq_len = seq_lengths[b]  # 실제 시퀀스 길이 사용
            
            for t in range(actual_seq_len):
                sar_idx = t * 3
                
                # State 임베딩 (패딩 필요시)
                state_emb = states[b, t]
                if state_emb.size(0) < d_model:
                    state_emb = F.pad(state_emb, (0, d_model - state_emb.size(0)))
                input_sequence[b, sar_idx] = state_emb[:d_model]
                
                # State 위치에서 다음 액션 예측 마스킹 (Decision Transformer 패턴)
                if t < actual_seq_len - 1:  # 마지막 State는 예측할 액션이 없음
                    action_prediction_mask[b, sar_idx] = True
                
                # Action 임베딩 (마지막 스텝 제외 - State 개수 = Action 개수 + 1)
                if t < actual_seq_len - 1:
                    action_emb = actions[b, t] if len(actions.shape) > 2 else torch.tensor([actions[b, t].item()])
                    if action_emb.size(0) < d_model:
                        action_emb = F.pad(action_emb, (0, d_model - action_emb.size(0)))
                    input_sequence[b, sar_idx + 1] = action_emb[:d_model]
                
                # RTG 임베딩 (모든 State에 대응)
                rtg_emb = torch.tensor([rtg[b, t].item()])
                if rtg_emb.size(0) < d_model:
                    rtg_emb = F.pad(rtg_emb, (0, d_model - rtg_emb.size(0)))
                input_sequence[b, sar_idx + 2] = rtg_emb[:d_model]
        
        # 어텐션 마스크를 실제 SAR 시퀀스 길이에 맞게 조정
        sar_attention_mask = torch.zeros(batch_size, max_sar_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            # 실제 유효한 토큰만 마스킹: State(seq_len) + Action(seq_len-1) + RTG(seq_len) = 3*seq_len - 1
            actual_sar_len = seq_len * 3 - 1  # 마지막 Action이 없으므로 -1
            sar_attention_mask[i, :actual_sar_len] = True
        
        # Causal mask 생성 (현재 위치 이후 모든 토큰 마스킹)
        causal_mask = torch.tril(torch.ones(max_sar_len, max_sar_len, dtype=torch.bool))
        
        # Padding mask와 causal mask 결합
        # sar_attention_mask: [batch_size, seq_len] - 패딩 마스크
        # causal_mask: [seq_len, seq_len] - 인과적 마스크
        combined_mask = sar_attention_mask.unsqueeze(1) & causal_mask.unsqueeze(0)
        
        return {
            'input_sequence': input_sequence,
            'attention_mask': combined_mask,  # Causal + Padding mask
            'action_prediction_mask': action_prediction_mask,
            'target_properties': target_properties,
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),
            # 원본 액션 데이터 추가 (손실 계산용)
            'actions': original_actions,
            'positions': original_positions,
            'parameters': original_parameters
        }
