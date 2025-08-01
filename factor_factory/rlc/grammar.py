
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
}
OPS = {k for k,(n,a) in TOKENS.items() if a==2}
TERMS = {k for k,(n,a) in TOKENS.items() if a==0}
ARITY = {k:a for k,(_,a) in TOKENS.items()}
N_TOKENS = len(TOKENS)

def name(tok:int) -> str:
    return TOKENS[tok][0]
