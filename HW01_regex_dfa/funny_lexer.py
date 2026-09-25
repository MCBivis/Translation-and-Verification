from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import json
import re
from typing import Optional


ASCII = tuple(chr(i) for i in range(128))


# ---------------------------------------------------------------------------
# Regular-expression AST
# Supported syntax:
#   literals, (...) , |, *, +, ?, character classes [...],
#   ranges [a-z], negated classes [^...], escapes \n \r \t \s \d \w and \xHH
# ---------------------------------------------------------------------------

@dataclass
class RNode:
    kind: str
    value: object = None
    left: Optional["RNode"] = None
    right: Optional["RNode"] = None


class RegexParser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def parse(self) -> RNode:
        node = self.parse_alt()
        if self.i != len(self.s):
            raise ValueError(f"unexpected character at {self.i}: {self.s[self.i]!r}")
        return node

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def take(self) -> str:
        c = self.peek()
        if c:
            self.i += 1
        return c

    def parse_alt(self) -> RNode:
        node = self.parse_concat()
        while self.peek() == "|":
            self.take()
            node = RNode("alt", left=node, right=self.parse_concat())
        return node

    def parse_concat(self) -> RNode:
        nodes = []
        while self.peek() and self.peek() not in ")|":
            nodes.append(self.parse_repeat())
        if not nodes:
            return RNode("epsilon")
        node = nodes[0]
        for nxt in nodes[1:]:
            node = RNode("concat", left=node, right=nxt)
        return node

    def parse_repeat(self) -> RNode:
        node = self.parse_atom()
        while self.peek() and self.peek() in "*+?":
            op = self.take()
            node = RNode(op, left=node)
        return node

    def parse_atom(self) -> RNode:
        c = self.take()
        if not c:
            raise ValueError("unexpected end of regex")
        if c == "(":
            node = self.parse_alt()
            if self.take() != ")":
                raise ValueError("missing ')'")
            return node
        if c == "[":
            return RNode("class", self.parse_class())
        if c == "\\":
            return RNode("class", self.parse_escape())
        return RNode("lit", c)

    def parse_escape(self) -> set[str]:
        c = self.take()
        if not c:
            raise ValueError("dangling escape")
        if c == "n":
            return {"\n"}
        if c == "r":
            return {"\r"}
        if c == "t":
            return {"\t"}
        if c == "s":
            return {" ", "\t", "\r", "\n", "\v", "\f"}
        if c == "d":
            return set("0123456789")
        if c == "w":
            return set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        if c == "x":
            h = self.s[self.i:self.i+2]
            if len(h) != 2 or not re.fullmatch(r"[0-9A-Fa-f]{2}", h):
                raise ValueError(r"expected two hex digits after \x")
            self.i += 2
            return {chr(int(h, 16))}
        return {c}

    def parse_class(self) -> set[str]:
        chars: set[str] = set()
        negated = False
        if self.peek() == "^":
            self.take()
            negated = True

        while True:
            if not self.peek():
                raise ValueError("unterminated character class")
            if self.peek() == "]":
                self.take()
                break

            if self.peek() == "\\":
                self.take()
                first_set = self.parse_escape()
            else:
                first_set = {self.take()}

            # Range is only meaningful for a single literal.
            if self.peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]" and len(first_set) == 1:
                self.take()
                if self.take() == "\\":
                    second_set = self.parse_escape()
                else:
                    second_set = {self.s[self.i - 1]}
                if len(second_set) != 1:
                    raise ValueError("range endpoint must be one character")
                a, b = ord(next(iter(first_set))), ord(next(iter(second_set)))
                if a > b:
                    raise ValueError("reversed range")
                chars.update(chr(x) for x in range(a, b + 1))
            else:
                chars.update(first_set)

        return (set(ASCII) - chars) if negated else chars


# ---------------------------------------------------------------------------
# Thompson NFA
# ---------------------------------------------------------------------------

@dataclass
class NFAState:
    eps: set[int] = field(default_factory=set)
    edges: dict[str, set[int]] = field(default_factory=dict)
    accepts: list[tuple[int, str, bool]] = field(default_factory=list)


class NFA:
    def __init__(self):
        self.states: list[NFAState] = []

    def new(self) -> int:
        self.states.append(NFAState())
        return len(self.states) - 1

    def edge(self, a: int, ch: str, b: int):
        self.states[a].edges.setdefault(ch, set()).add(b)

    def epsilon(self, a: int, b: int):
        self.states[a].eps.add(b)


def thompson(ast: RNode, nfa: NFA) -> tuple[int, int]:
    k = ast.kind
    if k == "epsilon":
        s, e = nfa.new(), nfa.new()
        nfa.epsilon(s, e)
        return s, e
    if k == "lit":
        s, e = nfa.new(), nfa.new()
        nfa.edge(s, ast.value, e)
        return s, e
    if k == "class":
        s, e = nfa.new(), nfa.new()
        for ch in ast.value:
            nfa.edge(s, ch, e)
        return s, e
    if k == "concat":
        a, b = thompson(ast.left, nfa)
        c, d = thompson(ast.right, nfa)
        nfa.epsilon(b, c)
        return a, d
    if k == "alt":
        s, e = nfa.new(), nfa.new()
        a, b = thompson(ast.left, nfa)
        c, d = thompson(ast.right, nfa)
        nfa.epsilon(s, a); nfa.epsilon(s, c)
        nfa.epsilon(b, e); nfa.epsilon(d, e)
        return s, e
    if k == "?":
        s, e = nfa.new(), nfa.new()
        a, b = thompson(ast.left, nfa)
        nfa.epsilon(s, e); nfa.epsilon(s, a); nfa.epsilon(b, e)
        return s, e
    if k == "*":
        s, e = nfa.new(), nfa.new()
        a, b = thompson(ast.left, nfa)
        nfa.epsilon(s, e); nfa.epsilon(s, a)
        nfa.epsilon(b, a); nfa.epsilon(b, e)
        return s, e
    if k == "+":
        a, b = thompson(ast.left, nfa)
        s, e = nfa.new(), nfa.new()
        nfa.epsilon(s, a)
        nfa.epsilon(b, a); nfa.epsilon(b, e)
        return s, e
    raise ValueError(f"unknown AST node {k}")


# ---------------------------------------------------------------------------
# Combined NFA -> DFA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenSpec:
    name: str
    regex: str
    skip: bool = False


@dataclass
class DFA:
    transitions: dict[int, dict[str, int]]
    start: int
    accepts: dict[int, tuple[str, bool]]
    alphabet: tuple[str, ...]
    trap: int

    @property
    def states(self) -> set[int]:
        out = {self.start, self.trap}
        out.update(self.transitions)
        for row in self.transitions.values():
            out.update(row.values())
        return out

    def run(self, text: str):
        state = self.start
        for ch in text:
            if ch not in ASCII:
                return None
            state = self.transitions[state][ch]
        return self.accepts.get(state)

    def tokenize(self, text: str):
        result = []
        i = 0
        while i < len(text):
            if ord(text[i]) >= 128:
                raise ValueError(f"non-ASCII character at {i}: {text[i]!r}")

            state = self.start
            last = None
            j = i
            while j < len(text) and ord(text[j]) < 128:
                state = self.transitions[state][text[j]]
                if state == self.trap:
                    break
                if state in self.accepts:
                    last = (j + 1, self.accepts[state])
                j += 1

            if last is None:
                raise ValueError(f"lexical error at {i}: {text[i]!r}")

            end, (name, skip) = last
            lexeme = text[i:end]
            if not skip:
                result.append((name, lexeme))
            i = end
        return result


def epsilon_closure(nfa: NFA, states: set[int]) -> frozenset[int]:
    stack = list(states)
    out = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.states[s].eps:
            if t not in out:
                out.add(t)
                stack.append(t)
    return frozenset(out)


def move(nfa: NFA, states: frozenset[int], ch: str) -> frozenset[int]:
    out = set()
    for s in states:
        out.update(nfa.states[s].edges.get(ch, ()))
    return epsilon_closure(nfa, out) if out else frozenset()


def build_combined_nfa(specs: list[TokenSpec]):
    nfa = NFA()
    start = nfa.new()

    for priority, spec in enumerate(specs):
        ast = RegexParser(spec.regex).parse()
        s, e = thompson(ast, nfa)
        nfa.epsilon(start, s)
        nfa.states[e].accepts.append((priority, spec.name, spec.skip))

    return nfa, start


def determinize(nfa: NFA, start: int, specs: list[TokenSpec]) -> DFA:
    dfa_states: list[frozenset[int]] = []
    index: dict[frozenset[int], int] = {}
    transitions: dict[int, dict[str, int]] = {}

    initial = epsilon_closure(nfa, {start})
    dfa_states.append(initial)
    index[initial] = 0
    q = deque([initial])

    while q:
        subset = q.popleft()
        sid = index[subset]
        transitions.setdefault(sid, {})
        for ch in ASCII:
            nxt = move(nfa, subset, ch)
            if not nxt:
                continue
            if nxt not in index:
                index[nxt] = len(dfa_states)
                dfa_states.append(nxt)
                q.append(nxt)
            transitions[sid][ch] = index[nxt]

    accepts: dict[int, tuple[str, bool]] = {}
    for sid, subset in enumerate(dfa_states):
        candidates = []
        for nfa_state in subset:
            candidates.extend(nfa.states[nfa_state].accepts)
        if candidates:
            priority, name, skip = min(candidates, key=lambda x: x[0])
            accepts[sid] = (name, skip)

    trap = len(dfa_states)
    transitions[trap] = {ch: trap for ch in ASCII}
    for sid in range(trap):
        for ch in ASCII:
            transitions[sid].setdefault(ch, trap)

    return DFA(transitions, 0, accepts, ASCII, trap)


# ---------------------------------------------------------------------------
# Hopcroft minimization for complete DFA
# ---------------------------------------------------------------------------

def minimize_hopcroft(dfa: DFA) -> DFA:
    states = set(dfa.states)
    alphabet = dfa.alphabet

    by_label: dict[object, set[int]] = {}
    for s in states:
        by_label.setdefault(dfa.accepts.get(s), set()).add(s)
    P = [b for b in by_label.values() if b]
    W = [b.copy() for b in P]

    inverse = {c: {} for c in alphabet}
    for p in states:
        for c in alphabet:
            q = dfa.transitions[p][c]
            inverse[c].setdefault(q, set()).add(p)

    while W:
        A = W.pop()
        for c in alphabet:
            X = set()
            for q in A:
                X.update(inverse[c].get(q, ()))
            if not X:
                continue

            newP = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if not inter or not diff:
                    newP.append(Y)
                    continue

                newP.extend((inter, diff))
                replaced = False
                for wi, Wg in enumerate(W):
                    if Wg == Y:
                        W[wi:wi + 1] = [inter, diff]
                        replaced = True
                        break
                if not replaced:
                    W.append(inter if len(inter) <= len(diff) else diff)
            P = newP

    block_of = {}
    for i, block in enumerate(P):
        for s in block:
            block_of[s] = i

    new_trans = {}
    new_accepts = {}
    for i, block in enumerate(P):
        rep = next(iter(block))
        new_trans[i] = {c: block_of[dfa.transitions[rep][c]] for c in alphabet}
        if rep in dfa.accepts:
            new_accepts[i] = dfa.accepts[rep]

    return DFA(
        transitions=new_trans,
        start=block_of[dfa.start],
        accepts=new_accepts,
        alphabet=alphabet,
        trap=block_of[dfa.trap],
    )


def build_lexer(specs: list[TokenSpec]):
    nfa, nfa_start = build_combined_nfa(specs)
    dfa = determinize(nfa, nfa_start, specs)
    minimized = minimize_hopcroft(dfa)
    return nfa, dfa, minimized


def export_dfa(dfa: DFA, path: str):
    data = {
        "alphabet": [ord(c) for c in dfa.alphabet],
        "start": dfa.start,
        "trap": dfa.trap,
        "accepting": {str(k): list(v) for k, v in dfa.accepts.items()},
        "transitions": {
            str(s): {str(ord(ch)): t for ch, t in row.items()}
            for s, row in sorted(dfa.transitions.items())
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Funny token specification
# ---------------------------------------------------------------------------

KEYWORDS = [
    "function", "returns", "requires", "ensures", "uses",
    "if", "else", "while", "invariant", "assert", "assume",
    "forall", "exists", "true", "false", "and", "or", "not",
    "int", "length",
]

TOKEN_SPECS = (
    [TokenSpec("KW_" + w.upper(), re.escape(w)) for w in KEYWORDS]
    + [
        TokenSpec("INT", r"(0|[1-9][0-9]*)"),
        TokenSpec("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
        TokenSpec("WS", r"[ \t\r\n]+", skip=True),
        TokenSpec("COMMENT", r"//[^\r\n]*", skip=True),
        TokenSpec("LPAREN", r"\("),
        TokenSpec("RPAREN", r"\)"),
        TokenSpec("LBRACKET", r"\["),
        TokenSpec("RBRACKET", r"\]"),
        TokenSpec("LBRACE", r"\{"),
        TokenSpec("RBRACE", r"\}"),
        TokenSpec("COMMA", r","),
        TokenSpec("SEMICOLON", r";"),
        TokenSpec("PLUS", r"\+"),
        TokenSpec("MINUS", r"-"),
        TokenSpec("STAR", r"\*"),
        TokenSpec("SLASH", r"/"),
        TokenSpec("EQ", r"=="),
        TokenSpec("NE", r"!="),
        TokenSpec("LE", r"<="),
        TokenSpec("GE", r">="),
        TokenSpec("LT", r"<"),
        TokenSpec("GT", r">"),
        TokenSpec("ASSIGN", r"="),
    ]
)


def run_builtin_tests(dfa: DFA):
    tests = [
        ("empty", "", []),
        ("spaces", " \t  \r\n", []),
        ("zero", "0", [("INT", "0")]),
        ("leading_zero", "01", [("INT", "0"), ("INT", "1")]),
        ("zero_and_number", "01002345", [("INT", "0"), ("INT", "1002345")]),
        ("identifier_underscore", "_abc A_1", [("IDENT", "_abc"), ("IDENT", "A_1")]),
        ("keywords", "function returns while if else assert assume invariant length",
         [(f"KW_{w.upper()}", w) for w in
          ["function", "returns", "while", "if", "else", "assert", "assume", "invariant", "length"]]),
        ("delimiters", "()[]{},;", [
            ("LPAREN","("),("RPAREN",")"),("LBRACKET","["),("RBRACKET","]"),
            ("LBRACE","{"),("RBRACE","}"),("COMMA",","),("SEMICOLON",";")]),
        ("operators", "+ - * / == != <= >= < > =", [
            ("PLUS","+"),("MINUS","-"),("STAR","*"),("SLASH","/"),
            ("EQ","=="),("NE","!="),("LE","<="),("GE",">="),("LT","<"),("GT",">"),("ASSIGN","=")]),
        ("comment", "// hello world\r\nx", [("IDENT","x")]),
        ("non_ascii", "привет", None),
        ("unknown_ascii", "@", None),
    ]

    report = []
    for name, text, expected in tests:
        try:
            actual = dfa.tokenize(text)
            ok = expected is not None and actual == expected
            status = "PASS" if ok else "FAIL"
            report.append({"name": name, "input": text, "expected": expected,
                           "actual": actual, "status": status})
        except ValueError as e:
            ok = expected is None or name == "non_ascii"
            report.append({"name": name, "input": text, "expected": expected,
                           "actual": f"ERROR: {e}", "status": "PASS" if ok else "FAIL"})
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="dfa.json")
    ap.add_argument("--tests", default="test_results.json")
    args = ap.parse_args()

    nfa, dfa, minimized = build_lexer(list(TOKEN_SPECS))
    export_dfa(minimized, args.export)

    report = run_builtin_tests(minimized)
    with open(args.tests, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"NFA states: {len(nfa.states)}")
    print(f"DFA states before minimization: {len(dfa.states)}")
    print(f"DFA states after Hopcroft minimization: {len(minimized.states)}")
    print(f"Trap state: {minimized.trap}")
    print(f"Tests: {sum(x['status']=='PASS' for x in report)}/{len(report)} PASS")
    print(f"Exported: {args.export}")
    print(f"Test report: {args.tests}")


if __name__ == "__main__":
    main()
