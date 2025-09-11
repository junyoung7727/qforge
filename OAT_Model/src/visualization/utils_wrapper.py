"""
utils_wrapper.py - Import 문제를 해결하기 위한 임시 래퍼 모듈

이 모듈은 src.utils.debug_utils를 utils.debug_utils로 매핑하여
decision_transformer.py의 import 문을 수정하지 않고도 사용할 수 있도록 합니다.
"""

import sys
from pathlib import Path

# src.utils를 utils로 매핑
sys.path.append(str(Path(__file__).parent.parent))
import utils.debug_utils

# 기존 모듈의 심볼들을 현재 네임스페이스에 노출
from utils.debug_utils import debug_print, debug_tensor_info

# utils.debug_utils를 sys.modules에 등록하여 다른 모듈에서 import할 수 있도록 함
sys.modules['utils.debug_utils'] = utils.debug_utils
