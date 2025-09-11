"""
RTG Cache Manager
RTG 계산 결과를 캐싱하여 재계산 방지
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib


class RTGCacheManager:
    """RTG 계산 결과 캐싱 관리자"""
    
    def __init__(self, cache_dir: str = "cache/rtg"):
        """
        Args:
            cache_dir: 캐시 파일 저장 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "rtg_cache.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # 캐시 로드
        self.cache_data = self._load_cache()
        self.metadata = self._load_metadata()
    
    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 캐시 파일 로드 실패: {e}")
                return {}
        return {}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 메타데이터 파일 로드 실패: {e}")
                return {}
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_circuits": 0,
            "cache_version": "1.0"
        }
    
    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 캐시 파일 저장 실패: {e}")
    
    def _save_metadata(self):
        """메타데이터 파일 저장"""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["total_circuits"] = len(self.cache_data)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 메타데이터 파일 저장 실패: {e}")
    
    def _generate_circuit_hash(self, circuit_spec: Dict[str, Any]) -> str:
        """회로 스펙의 해시 생성 (중복 감지용)"""
        # 회로의 핵심 정보만 해시화
        circuit_key = {
            'num_qubits': circuit_spec.get('num_qubits', 0),
            'gates': [
                {
                    'name': gate.get('name', ''),
                    'qubits': gate.get('qubits', []),
                    'parameters': gate.get('parameters', [])
                }
                for gate in circuit_spec.get('gates', [])
            ]
        }
        circuit_str = json.dumps(circuit_key, sort_keys=True)
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    def get_cached_rtg(self, circuit_id: str) -> Optional[Dict[str, Any]]:
        """캐시된 RTG 데이터 조회"""
        return self.cache_data.get(circuit_id)
    
    def cache_rtg_result(self, circuit_id: str, circuit_spec: Dict[str, Any], 
                        rtg_values: List[float], rewards: List[float], 
                        properties: List[Dict[str, float]], target_properties: Dict[str, float]):
        """RTG 계산 결과 캐시"""
        circuit_hash = self._generate_circuit_hash(circuit_spec)
        
        cache_entry = {
            'circuit_id': circuit_id,
            'circuit_hash': circuit_hash,
            'rtg_values': rtg_values,
            'rewards': rewards,
            'properties': properties,
            'target_properties': target_properties,
            'cached_at': datetime.now().isoformat(),
            'num_gates': len(circuit_spec.get('gates', []))
        }
        
        self.cache_data[circuit_id] = cache_entry
    
    def has_cached_rtg(self, circuit_id: str) -> bool:
        """RTG 캐시 존재 여부 확인"""
        return circuit_id in self.cache_data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        if not self.cache_data:
            return {
                'total_cached': 0,
                'cache_hit_rate': 0.0,
                'avg_gates_per_circuit': 0.0
            }
        
        total_gates = sum(entry.get('num_gates', 0) for entry in self.cache_data.values())
        avg_gates = total_gates / len(self.cache_data) if self.cache_data else 0
        
        return {
            'total_cached': len(self.cache_data),
            'avg_gates_per_circuit': avg_gates,
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            'last_updated': self.metadata.get('last_updated', 'Unknown')
        }
    
    def save_all(self):
        """모든 캐시 데이터 저장"""
        self._save_cache()
        self._save_metadata()
        
    def clear_cache(self):
        """캐시 초기화"""
        self.cache_data.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("🗑️ RTG 캐시가 초기화되었습니다.")
    
    def batch_cache_rtg_results(self, results: List[Dict[str, Any]]):
        """배치로 RTG 결과 캐시"""
        for result in results:
            self.cache_rtg_result(
                circuit_id=result['circuit_id'],
                circuit_spec=result['circuit_spec'],
                rtg_values=result['rtg_values'],
                rewards=result['rewards'],
                properties=result['properties'],
                target_properties=result['target_properties']
            )
        
        # 배치 저장
        self.save_all()
        print(f"💾 {len(results)}개 RTG 결과가 캐시되었습니다.")
