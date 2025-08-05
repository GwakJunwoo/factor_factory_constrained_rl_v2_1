
from __future__ import annotations

# Token dictionary: id -> (name, arity)
TOKENS = {
    0: ("ADD", 2),
    1: ("SUB", 2),
    2: ("MUL", 2),
    3: ("DIV", 2),
    4: ("CLOSE", 0),
    5: ("OPEN", 0),
    6: ("HIGH", 0),
    7: ("LOW", 0),
    8: ("VOLUME", 0),
    9: ("SMA10", 0),
    10: ("SMA20", 0),
    11: ("RSI14", 0),
    12: ("CONST1", 0),
    13: ("SMA5", 0),
    14: ("EMA10", 0),
    15: ("EMA20", 0),
    16: ("BBANDS_UPPER", 0),
    17: ("BBANDS_LOWER", 0),
    18: ("MACD", 0),
    19: ("STOCH", 0),
    20: ("MAX", 2),
    21: ("MIN", 2),
    22: ("ABS", 1),
    23: ("LOG", 1),
    24: ("LAG1", 1),
}
OPS = {k for k,(n,a) in TOKENS.items() if a>=1}  # 단항/이진 연산자 모두 포함
BINARY_OPS = {k for k,(n,a) in TOKENS.items() if a==2}  # 이진 연산자만
UNARY_OPS = {k for k,(n,a) in TOKENS.items() if a==1}   # 단항 연산자만
TERMS = {k for k,(n,a) in TOKENS.items() if a==0}
ARITY = {k:a for k,(_,a) in TOKENS.items()}
N_TOKENS = len(TOKENS)

TOKEN_NAMES = {k: v[0] for k, v in TOKENS.items()}

def name(tok:int) -> str:
    return TOKENS[tok][0]
