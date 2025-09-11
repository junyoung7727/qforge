"""
Episode-wise Gate Reward Visualization Module
각 에피소드별 게이트 추가시마다 받는 리워드 시각화
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from rtg.core.rtg_calculator import RTGCalculator


class EpisodeRewardVisualizer:
    """에피소드별 게이트 리워드 시각화 클래스"""
    
    def __init__(self, save_dir: str = "reward_visualization"):
        """
        Args:
            save_dir: 시각화 결과 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def visualize_episode_rewards(self,
                                episode_data: Dict[str, Any],
                                episode_id: str = "episode_1") -> None:
        """
        단일 에피소드의 게이트별 리워드 시각화
        
        Args:
            episode_data: RTG 계산기에서 반환된 에피소드 데이터
            episode_id: 에피소드 식별자
        """
        rtg_values = episode_data.get('rtg_values', [])
        rewards = episode_data.get('rewards', [])
        properties = episode_data.get('properties', [])
        target_props = episode_data.get('target_properties', {})
        
        if not rtg_values or not rewards:
            print(f"⚠️ {episode_id}: 시각화할 데이터가 없습니다")
            return
        
        # 1. 리워드 및 RTG 시계열 플롯
        self._plot_reward_timeline(rtg_values, rewards, episode_id)
        
        # 2. 속성별 진화 플롯
        self._plot_property_evolution(properties, target_props, episode_id)
        
        # 3. 게이트별 리워드 히트맵
        self._plot_gate_reward_heatmap(rewards, episode_id)
        
        # 4. 인터랙티브 대시보드
        self._create_interactive_dashboard(episode_data, episode_id)
        
        print(f"✅ {episode_id} 시각화 완료: {self.save_dir}")
    
    def _plot_reward_timeline(self, 
                            rtg_values: List[float], 
                            rewards: List[float],
                            episode_id: str) -> None:
        """리워드 및 RTG 시계열 플롯"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        steps = range(len(rewards))
        
        # 상단: 스텝별 리워드
        ax1.plot(steps, rewards, 'o-', linewidth=2, markersize=6, 
                color='#2E86AB', label='Step Reward')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f'{episode_id}: Gate-wise Rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Gate Step')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 하단: RTG 값
        ax2.plot(steps, rtg_values, 's-', linewidth=2, markersize=6,
                color='#A23B72', label='Return-to-Go')
        ax2.set_title(f'{episode_id}: Return-to-Go Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Gate Step')
        ax2.set_ylabel('RTG')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{episode_id}_reward_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_property_evolution(self,
                               properties: List[Dict[str, float]],
                               target_props: Dict[str, float],
                               episode_id: str) -> None:
        """속성별 진화 플롯"""
        if not properties:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        steps = range(len(properties))
        
        prop_names = ['entanglement', 'expressibility', 'fidelity']
        colors = ['#F18F01', '#C73E1D', '#2E86AB']
        
        for i, prop_name in enumerate(prop_names):
            ax = axes[i]
            
            # 예측값 진화
            prop_values = [p.get(prop_name, 0) for p in properties]
            ax.plot(steps, prop_values, 'o-', linewidth=2, markersize=6,
                   color=colors[i], label=f'Predicted {prop_name.title()}')
            
            # 타겟값 라인
            if prop_name in target_props:
                target_val = target_props[prop_name]
                ax.axhline(y=target_val, color=colors[i], linestyle='--', 
                          alpha=0.7, label=f'Target ({target_val:.3f})')
            
            ax.set_title(f'{prop_name.title()} Evolution', fontweight='bold')
            ax.set_xlabel('Gate Step')
            ax.set_ylabel(prop_name.title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'{episode_id}: Property Evolution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{episode_id}_property_evolution.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gate_reward_heatmap(self,
                                rewards: List[float],
                                episode_id: str) -> None:
        """게이트별 리워드 히트맵"""
        if len(rewards) < 2:
            return
        
        # 리워드를 2D 그리드로 변환 (적절한 크기로)
        grid_size = int(np.ceil(np.sqrt(len(rewards))))
        padded_rewards = rewards + [0] * (grid_size * grid_size - len(rewards))
        reward_grid = np.array(padded_rewards).reshape(grid_size, grid_size)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 히트맵 생성
        im = ax.imshow(reward_grid, cmap='RdYlBu_r', aspect='auto')
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Reward', rotation=270, labelpad=20)
        
        # 텍스트 어노테이션
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(rewards):
                    text = ax.text(j, i, f'{rewards[idx]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'{episode_id}: Gate Reward Heatmap', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Gate Position (X)')
        ax.set_ylabel('Gate Position (Y)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{episode_id}_reward_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_dashboard(self,
                                    episode_data: Dict[str, Any],
                                    episode_id: str) -> None:
        """인터랙티브 대시보드 생성"""
        rtg_values = episode_data.get('rtg_values', [])
        rewards = episode_data.get('rewards', [])
        properties = episode_data.get('properties', [])
        target_props = episode_data.get('target_properties', {})
        
        if not rtg_values:
            return
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Step Rewards', 'Return-to-Go', 
                          'Property Evolution', 'Reward Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"type": "histogram"}]]
        )
        
        steps = list(range(len(rewards)))
        
        # 1. 스텝별 리워드
        fig.add_trace(
            go.Scatter(x=steps, y=rewards, mode='lines+markers',
                      name='Step Reward', line=dict(color='#2E86AB')),
            row=1, col=1
        )
        
        # 2. RTG 값
        fig.add_trace(
            go.Scatter(x=steps, y=rtg_values, mode='lines+markers',
                      name='RTG', line=dict(color='#A23B72')),
            row=1, col=2
        )
        
        # 3. 속성 진화
        if properties:
            for prop_name, color in zip(['entanglement', 'expressibility', 'fidelity'],
                                      ['#F18F01', '#C73E1D', '#2E86AB']):
                prop_values = [p.get(prop_name, 0) for p in properties]
                fig.add_trace(
                    go.Scatter(x=steps, y=prop_values, mode='lines+markers',
                              name=f'{prop_name.title()}', line=dict(color=color)),
                    row=2, col=1
                )
                
                # 타겟값 라인
                if prop_name in target_props:
                    fig.add_hline(y=target_props[prop_name], 
                                line_dash="dash", line_color=color,
                                row=2, col=1)
        
        # 4. 리워드 분포
        fig.add_trace(
            go.Histogram(x=rewards, name='Reward Distribution',
                        marker_color='#F18F01', opacity=0.7),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=f'{episode_id}: Interactive Reward Dashboard',
            showlegend=True,
            height=800
        )
        
        # HTML로 저장
        fig.write_html(self.save_dir / f'{episode_id}_dashboard.html')
    
    def compare_episodes(self,
                        episodes_data: Dict[str, Dict[str, Any]],
                        comparison_name: str = "episode_comparison") -> None:
        """
        여러 에피소드 비교 시각화
        
        Args:
            episodes_data: 에피소드별 데이터 딕셔너리
            comparison_name: 비교 시각화 이름
        """
        if len(episodes_data) < 2:
            print("⚠️ 비교할 에피소드가 부족합니다")
            return
        
        # 1. 에피소드별 총 리워드 비교
        self._plot_episode_total_rewards(episodes_data, comparison_name)
        
        # 2. 에피소드별 RTG 진화 비교
        self._plot_episode_rtg_comparison(episodes_data, comparison_name)
        
        # 3. 속성 수렴 비교
        self._plot_property_convergence_comparison(episodes_data, comparison_name)
        
        print(f"✅ 에피소드 비교 시각화 완료: {comparison_name}")
    
    def _plot_episode_total_rewards(self,
                                  episodes_data: Dict[str, Dict[str, Any]],
                                  comparison_name: str) -> None:
        """에피소드별 총 리워드 비교"""
        episode_names = []
        total_rewards = []
        
        for ep_name, ep_data in episodes_data.items():
            rewards = ep_data.get('rewards', [])
            if rewards:
                episode_names.append(ep_name)
                total_rewards.append(sum(rewards))
        
        if not total_rewards:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(episode_names, total_rewards, 
                     color=sns.color_palette("husl", len(episode_names)))
        
        # 값 표시
        for bar, reward in zip(bars, total_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{reward:.3f}', ha='center', va='bottom')
        
        ax.set_title('Episode Total Rewards Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{comparison_name}_total_rewards.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_rtg_comparison(self,
                                   episodes_data: Dict[str, Dict[str, Any]],
                                   comparison_name: str) -> None:
        """에피소드별 RTG 진화 비교"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = sns.color_palette("husl", len(episodes_data))
        
        for i, (ep_name, ep_data) in enumerate(episodes_data.items()):
            rtg_values = ep_data.get('rtg_values', [])
            if rtg_values:
                steps = range(len(rtg_values))
                ax.plot(steps, rtg_values, 'o-', linewidth=2, markersize=4,
                       color=colors[i], label=ep_name, alpha=0.8)
        
        ax.set_title('Episode RTG Evolution Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Gate Step')
        ax.set_ylabel('Return-to-Go')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{comparison_name}_rtg_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_property_convergence_comparison(self,
                                            episodes_data: Dict[str, Dict[str, Any]],
                                            comparison_name: str) -> None:
        """속성 수렴 비교"""
        prop_names = ['entanglement', 'expressibility', 'fidelity']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = sns.color_palette("husl", len(episodes_data))
        
        for prop_idx, prop_name in enumerate(prop_names):
            ax = axes[prop_idx]
            
            for ep_idx, (ep_name, ep_data) in enumerate(episodes_data.items()):
                properties = ep_data.get('properties', [])
                target_props = ep_data.get('target_properties', {})
                
                if properties:
                    prop_values = [p.get(prop_name, 0) for p in properties]
                    steps = range(len(prop_values))
                    ax.plot(steps, prop_values, 'o-', linewidth=2, markersize=4,
                           color=colors[ep_idx], label=ep_name, alpha=0.8)
                
                # 타겟값 (첫 번째 에피소드 기준)
                if ep_idx == 0 and prop_name in target_props:
                    target_val = target_props[prop_name]
                    ax.axhline(y=target_val, color='red', linestyle='--',
                              alpha=0.7, label=f'Target ({target_val:.3f})')
            
            ax.set_title(f'{prop_name.title()} Convergence', fontweight='bold')
            ax.set_xlabel('Gate Step')
            ax.set_ylabel(prop_name.title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'{comparison_name}: Property Convergence Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{comparison_name}_property_convergence.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self,
                            episodes_data: Dict[str, Dict[str, Any]],
                            report_name: str = "reward_analysis_report") -> None:
        """
        리워드 분석 요약 보고서 생성
        
        Args:
            episodes_data: 에피소드별 데이터
            report_name: 보고서 이름
        """
        # 통계 계산
        stats = {}
        for ep_name, ep_data in episodes_data.items():
            rewards = ep_data.get('rewards', [])
            rtg_values = ep_data.get('rtg_values', [])
            
            if rewards and rtg_values:
                stats[ep_name] = {
                    'total_reward': sum(rewards),
                    'avg_reward': np.mean(rewards),
                    'max_reward': max(rewards),
                    'min_reward': min(rewards),
                    'final_rtg': rtg_values[-1] if rtg_values else 0,
                    'rtg_improvement': rtg_values[-1] - rtg_values[0] if len(rtg_values) > 1 else 0
                }
        
        # 보고서 생성
        report_path = self.save_dir / f'{report_name}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EPISODE REWARD ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            for ep_name, ep_stats in stats.items():
                f.write(f"Episode: {ep_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Reward: {ep_stats['total_reward']:.4f}\n")
                f.write(f"Average Reward: {ep_stats['avg_reward']:.4f}\n")
                f.write(f"Max Reward: {ep_stats['max_reward']:.4f}\n")
                f.write(f"Min Reward: {ep_stats['min_reward']:.4f}\n")
                f.write(f"Final RTG: {ep_stats['final_rtg']:.4f}\n")
                f.write(f"RTG Improvement: {ep_stats['rtg_improvement']:.4f}\n")
                f.write("\n")
        
        print(f"✅ 요약 보고서 생성: {report_path}")


def create_episode_visualizer(save_dir: str = "reward_visualization") -> EpisodeRewardVisualizer:
    """
    에피소드 리워드 시각화기 생성 팩토리 함수
    
    Args:
        save_dir: 저장 디렉토리
        
    Returns:
        설정된 시각화기
    """
    return EpisodeRewardVisualizer(save_dir)


if __name__ == "__main__":
    # 테스트 코드
    print("🎨 Episode Reward Visualizer 테스트 시작")
    
    visualizer = create_episode_visualizer("test_visualization")
    
    # 더미 데이터 생성
    dummy_episode = {
        'rtg_values': [1.5, 1.2, 0.8, 0.5, 0.2],
        'rewards': [0.3, -0.1, 0.2, -0.2, 0.1],
        'properties': [
            {'entanglement': 0.5, 'expressibility': 0.6, 'fidelity': 0.8},
            {'entanglement': 0.6, 'expressibility': 0.65, 'fidelity': 0.82},
            {'entanglement': 0.7, 'expressibility': 0.7, 'fidelity': 0.85},
            {'entanglement': 0.75, 'expressibility': 0.72, 'fidelity': 0.87},
            {'entanglement': 0.8, 'expressibility': 0.75, 'fidelity': 0.9}
        ],
        'target_properties': {'entanglement': 0.8, 'expressibility': 0.75, 'fidelity': 0.9}
    }
    
    visualizer.visualize_episode_rewards(dummy_episode, "test_episode")
    print("✅ 테스트 완료")
