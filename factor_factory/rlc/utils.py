
from __future__ import annotations
from .grammar import TOKENS, ARITY, name, OPS, TERMS

def tokens_to_infix(tokens: list[int]) -> str:
    """Convert prefix tokens to infix string for readability."""
    def rec(it):
        tok = next(it)
        nm = name(tok)
        if ARITY[tok]==0:
            return nm
        a = rec(it); b = rec(it)
        op = {0:"+",1:"-",2:"*",3:"/"}[tok]
        return f"({a} {op} {b})"
    return rec(iter(tokens))
