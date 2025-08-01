
import numpy as np
from .grammar import ARITY
def calc_tree_depth(tokens: list[int]) -> int:
    """
    prefix 토큰 리스트의 최대 트리 깊이 계산
    """
    idx = 0

    def _rec(depth: int = 1) -> int:
        nonlocal idx
        tok = tokens[idx]
        idx += 1
        if ARITY[tok] == 0:          # 터미널
            return depth
        # 이진/단항 등 자식 탐색
        depths = [_rec(depth + 1) for _ in range(ARITY[tok])]
        return max(depths)

    return _rec()

def count_entries(sig):
    sig = np.asarray(sig,dtype=int)
    prev=np.concatenate(([0],sig[:-1]))
    return int(np.sum((prev==0)&(sig!=0)))

def tokens_to_infix(tokens: list[int]) -> str:
    """트리 구조 토큰을 중위표기식 문자열로 변환"""
    from .grammar import TOKEN_NAMES, ARITY

    def helper(idx):
        tok = tokens[idx]
        name = TOKEN_NAMES[tok]
        if ARITY[tok] == 0:
            return name, idx + 1
        elif ARITY[tok] == 1:
            a, next_idx = helper(idx + 1)
            return f"{name}({a})", next_idx
        elif ARITY[tok] == 2:
            a, mid = helper(idx + 1)
            b, next_idx = helper(mid)
            return f"({a} {name} {b})", next_idx
        else:
            raise ValueError(f"Unknown arity for token {tok}")

    expr, _ = helper(0)
    return expr
