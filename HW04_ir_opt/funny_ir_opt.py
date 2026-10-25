from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import funny_parser as ast
from funny_lexer import TOKEN_SPECS, build_lexer
from funny_semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# IR format
# ---------------------------------------------------------------------------
# Each instruction is a dictionary with fields:
#   op: operation name; dst: destination variable/temp or null;
#   args: operands (variable names, integer/boolean literals, or labels);
#   label: optional label attached to the instruction.
# The supported operations include const, mov, arithmetic/comparison/logical ops,
# load/store, call, call_multi, jmp, br, assert, assume, return, and nop.
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    op: str
    dst: str | None = None
    args: list[Any] | None = None
    label: str | None = None
    line: int | None = None

    def __post_init__(self):
        if self.args is None:
            self.args = []

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if result["label"] is None:
            result.pop("label")
        if result["line"] is None:
            result.pop("line")
        return result

    def __str__(self) -> str:
        prefix = f"{self.label}: " if self.label else ""
        a = self.args or []
        if self.op == "const": body = f"{self.dst} = {a[0]}"
        elif self.op == "mov": body = f"{self.dst} = {a[0]}"
        elif self.op in {"add", "sub", "mul", "div", "and", "or", "eq", "ne", "lt", "le", "gt", "ge"}:
            symbol = {"add":"+", "sub":"-", "mul":"*", "div":"/", "and":"and", "or":"or", "eq":"==", "ne":"!=", "lt":"<", "le":"<=", "gt":">", "ge":">="}[self.op]
            body = f"{self.dst} = {a[0]} {symbol} {a[1]}"
        elif self.op == "neg": body = f"{self.dst} = -{a[0]}"
        elif self.op == "not": body = f"{self.dst} = not {a[0]}"
        elif self.op == "load": body = f"{self.dst} = {a[0]}[{a[1]}]"
        elif self.op == "store": body = f"{a[0]}[{a[1]}] = {a[2]}"
        elif self.op == "call": body = f"{self.dst} = call {a[0]}({', '.join(map(str, a[1:]))})"
        elif self.op == "call_multi": body = f"{', '.join(self.dst or [])} = call {a[0]}({', '.join(map(str, a[1:]))})"
        elif self.op == "jmp": body = f"goto {a[0]}"
        elif self.op == "br": body = f"if {a[0]} goto {a[1]} else goto {a[2]}"
        elif self.op == "assert": body = f"assert {a[0]}"
        elif self.op == "assume": body = f"assume {a[0]}"
        elif self.op == "return": body = "return" + (" " + ", ".join(map(str, a)) if a else "")
        else: body = self.op + (" " + ", ".join(map(str, a)) if a else "")
        return prefix + body


@dataclass
class BasicBlock:
    name: str
    instructions: list[Instruction]
    successors: list[str]
    predecessors: list[str]

    def to_dict(self):
        return {"name": self.name, "instructions": [i.to_dict() for i in self.instructions],
                "successors": self.successors, "predecessors": self.predecessors}


class IRGenerator:
    def __init__(self):
        self.instructions: list[Instruction] = []
        self.temp_counter = 0
        self.label_counter = 0
        self.current_function = ""

    def temp(self) -> str:
        self.temp_counter += 1
        return f"%t{self.temp_counter}"

    def label(self, prefix="L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"

    def emit(self, op, dst=None, args=None, node=None):
        self.instructions.append(Instruction(op, dst, args or [], line=getattr(node, "line", None)))

    def mark(self, label):
        self.instructions.append(Instruction("nop", args=[], label=label))

    def generate(self, program) -> dict[str, Any]:
        functions = []
        for fn in program.functions:
            self.instructions = []
            self.current_function = fn.name
            self.temp_counter = 0
            self.label_counter = 0
            entry = self.label("entry_")
            self.mark(entry)
            # Contracts are represented explicitly in IR for later verification/codegen.
            if fn.requires is not None:
                pred = self.predicate(fn.requires)
                self.emit("assume", args=[pred], node=fn.requires)
            self.statement(fn.body)
            if fn.ensures is not None:
                pred = self.predicate(fn.ensures)
                self.emit("assert", args=[pred], node=fn.ensures)
            self.emit("return", args=[v.name for v in fn.returns], node=fn)
            functions.append({"name": fn.name, "parameters": [v.name for v in fn.parameters],
                              "returns": [v.name for v in fn.returns],
                              "instructions": [i.to_dict() for i in self.instructions]})
        return {"format": "Funny-TAC-v1", "functions": functions}

    def statement(self, stmt):
        if stmt is None: return
        k = type(stmt).__name__
        if k == "Block":
            for child in stmt.statements: self.statement(child)
        elif k == "Assignment":
            value = stmt.value
            if type(value).__name__ == "FunctionCall" and value.name != "length":
                targets = [self.target_name(t) for t in stmt.targets]
                args = [self.expression(a) for a in value.arguments]
                if len(targets) > 1:
                    self.emit("call_multi", dst=targets, args=[value.name] + args, node=stmt)
                else:
                    result = self.temp()
                    self.emit("call", dst=result, args=[value.name] + args, node=value)
                    self.assign_target(stmt.targets[0], result, stmt)
            else:
                result = self.expression(value)
                for target in stmt.targets: self.assign_target(target, result, stmt)
        elif k == "IfStatement":
            then_label, else_label, end_label = self.label("if_then_"), self.label("if_else_"), self.label("if_end_")
            self.branch_predicate(stmt.condition, then_label, else_label)
            self.mark(then_label); self.statement(stmt.then_branch); self.emit("jmp", args=[end_label], node=stmt)
            self.mark(else_label)
            if stmt.else_branch is not None: self.statement(stmt.else_branch)
            self.emit("jmp", args=[end_label], node=stmt)
            self.mark(end_label)
        elif k == "WhileStatement":
            cond_label, body_label, end_label = self.label("while_cond_"), self.label("while_body_"), self.label("while_end_")
            self.mark(cond_label)
            if stmt.invariant is not None: self.emit("assert", args=[self.predicate(stmt.invariant)], node=stmt.invariant)
            self.branch_predicate(stmt.condition, body_label, end_label)
            self.mark(body_label); self.statement(stmt.body); self.emit("jmp", args=[cond_label], node=stmt)
            self.mark(end_label)
        elif k == "AssertStatement": self.emit("assert", args=[self.predicate(stmt.predicate)], node=stmt)
        elif k == "AssumeStatement": self.emit("assume", args=[self.predicate(stmt.predicate)], node=stmt)
        else: raise ValueError(f"IR generation does not support statement {k}")

    def target_name(self, target):
        k = type(target).__name__
        if k == "Variable": return target.name
        if k == "ArrayAccess": return target.array
        raise ValueError(f"unsupported assignment target {k}")

    def assign_target(self, target, value, node=None):
        if type(target).__name__ == "Variable": self.emit("mov", dst=target.name, args=[value], node=node)
        elif type(target).__name__ == "ArrayAccess":
            index = self.expression(target.index)
            self.emit("store", args=[target.array, index, value], node=node)
        else: raise ValueError(f"unsupported assignment target {type(target).__name__}")

    def expression(self, expr):
        k = type(expr).__name__
        if k == "IntLiteral": return expr.value
        if k == "BoolLiteral": return bool(expr.value)
        if k == "Variable": return expr.name
        if k == "ArrayAccess":
            index = self.expression(expr.index); dst = self.temp(); self.emit("load", dst=dst, args=[expr.array, index], node=expr); return dst
        if k == "FunctionCall":
            args = [self.expression(a) for a in expr.arguments]
            dst = self.temp()
            if expr.name == "length": self.emit("length", dst=dst, args=args, node=expr)
            else: self.emit("call", dst=dst, args=[expr.name] + args, node=expr)
            return dst
        if k == "UnaryOp":
            operand = self.expression(expr.operand); dst = self.temp()
            self.emit("neg" if expr.operator == "-" else "not", dst=dst, args=[operand], node=expr); return dst
        if k == "BinaryOp":
            left, right = self.expression(expr.left), self.expression(expr.right); dst = self.temp()
            op = {"+":"add", "-":"sub", "*":"mul", "/":"div", "and":"and", "or":"or", "->":"implies"}.get(expr.operator)
            if op is None: raise ValueError(f"unsupported binary operator {expr.operator}")
            self.emit(op, dst=dst, args=[left, right], node=expr); return dst
        if k in ("Comparison", "PredicateBinary", "PredicateNot", "BoolLiteral", "Quantifier", "FormulaRef"):
            dst = self.temp(); self.emit("mov", dst=dst, args=[self.predicate(expr)], node=expr); return dst
        raise ValueError(f"unsupported expression node {k}")

    def predicate(self, pred):
        k = type(pred).__name__
        if k == "BoolLiteral": return bool(pred.value)
        if k == "Comparison":
            left, right = self.expression(pred.left), self.expression(pred.right)
            op = {"==":"eq", "!=":"ne", "<":"lt", "<=":"le", ">":"gt", ">=":"ge"}.get(pred.operator)
            if op is None: raise ValueError(f"unsupported comparison {pred.operator}")
            dst = self.temp(); self.emit(op, dst=dst, args=[left, right], node=pred); return dst
        if k == "PredicateNot":
            value = self.predicate(pred.operand); dst = self.temp(); self.emit("not", dst=dst, args=[value], node=pred); return dst
        if k == "PredicateBinary":
            left, right = self.predicate(pred.left), self.predicate(pred.right)
            op = {"and":"and", "or":"or", "->":"implies"}.get(pred.operator, pred.operator)
            dst = self.temp(); self.emit(op, dst=dst, args=[left, right], node=pred); return dst
        if k == "Quantifier":
            # Quantifiers are preserved as a high-level operand until a later lowering pass.
            return {"quantifier": pred.kind, "variable": pred.variable.name, "body": self.predicate(pred.body)}
        if k == "FormulaRef": return {"formula": pred.name, "arguments": [self.expression(a) for a in pred.arguments]}
        if k in ("IntLiteral", "Variable", "BinaryOp", "UnaryOp", "ArrayAccess", "FunctionCall"):
            return self.expression(pred)
        raise ValueError(f"unsupported predicate node {k}")

    def branch_predicate(self, pred, true_label, false_label):
        # Preserve short-circuit semantics for logical conjunction/disjunction.
        if type(pred).__name__ == "PredicateNot":
            self.branch_predicate(pred.operand, false_label, true_label); return
        if type(pred).__name__ == "PredicateBinary" and pred.operator == "and":
            next_label = self.label("and_rhs_")
            self.branch_predicate(pred.left, next_label, false_label)
            self.mark(next_label); self.branch_predicate(pred.right, true_label, false_label); return
        if type(pred).__name__ == "PredicateBinary" and pred.operator == "or":
            next_label = self.label("or_rhs_")
            self.branch_predicate(pred.left, true_label, next_label)
            self.mark(next_label); self.branch_predicate(pred.right, true_label, false_label); return
        value = self.predicate(pred)
        self.emit("br", args=[value, true_label, false_label], node=pred)


# ---------------------------------------------------------------------------
# Basic blocks and control-flow graph
# ---------------------------------------------------------------------------

def build_cfg(function: dict[str, Any]) -> list[BasicBlock]:
    raw = [Instruction(**x) for x in function["instructions"]]
    if not raw: return []
    labels = {i.label: n for n, i in enumerate(raw) if i.label}
    leaders = {0}
    for n, ins in enumerate(raw):
        if ins.label: leaders.add(n)
        if ins.op in ("jmp", "br", "return") and n + 1 < len(raw): leaders.add(n + 1)
        if ins.op == "jmp":
            if ins.args[0] in labels: leaders.add(labels[ins.args[0]])
        elif ins.op == "br":
            for target in ins.args[1:3]:
                if target in labels: leaders.add(labels[target])
    starts = sorted(leaders)
    blocks, instruction_to_block = [], {}
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(raw)
        chunk = raw[start:end]
        label = next((i.label for i in chunk if i.label), f"B{idx}")
        # Labels are block names; unlabeled blocks receive stable names.
        block = BasicBlock(label, chunk, [], [])
        blocks.append(block)
        for n in range(start, end): instruction_to_block[n] = label
    label_to_block = {b.name: b for b in blocks}
    for idx, block in enumerate(blocks):
        last = block.instructions[-1] if block.instructions else None
        targets = []
        if last and last.op == "jmp": targets = [last.args[0]]
        elif last and last.op == "br": targets = [last.args[1], last.args[2]]
        elif last and last.op != "return" and idx + 1 < len(blocks): targets = [blocks[idx + 1].name]
        else: targets = []
        block.successors = list(dict.fromkeys(t for t in targets if t in label_to_block))
    for block in blocks:
        for successor in block.successors:
            label_to_block[successor].predecessors.append(block.name)
    return blocks


def cfg_to_dot(function_name: str, blocks: list[BasicBlock]) -> str:
    lines = ["digraph CFG {", "  node [shape=box,fontname=Courier];"]
    for b in blocks:
        label = "\\l".join([b.name] + [str(i) for i in b.instructions]) + "\\l"
        lines.append(f'  "{b.name}" [label="{label}"];')
        for s in b.successors: lines.append(f'  "{b.name}" -> "{s}";')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local optimizations
# ---------------------------------------------------------------------------

PURE_OPS = {"const", "mov", "add", "sub", "mul", "div", "neg", "not", "and", "or", "implies", "eq", "ne", "lt", "le", "gt", "ge", "length", "load"}
BINARY = {"add", "sub", "mul", "div", "and", "or", "implies", "eq", "ne", "lt", "le", "gt", "ge"}


def is_var(x): return isinstance(x, str) and (x.startswith("%t") or (x and (x[0].isalpha() or x[0] == "_")))


def eval_op(op, args):
    # IR variable names are represented as strings. Python would otherwise
    # happily evaluate expressions such as "r" + "i" or "r" == "%t4",
    # producing bogus constants. Only fold when every operand is a literal.
    if any(is_var(value) or isinstance(value, (dict, list)) for value in args):
        return None
    a = args[0] if args else None
    b = args[1] if len(args) > 1 else None
    try:
        if op == "add": return a + b
        if op == "sub": return a - b
        if op == "mul": return a * b
        if op == "div": return int(a / b)  # Funny integer arithmetic
        if op == "neg": return -a
        if op == "not": return not a
        if op == "and": return bool(a and b)
        if op == "or": return bool(a or b)
        if op == "implies": return (not bool(a)) or bool(b)
        if op == "eq": return a == b
        if op == "ne": return a != b
        if op == "lt": return a < b
        if op == "le": return a <= b
        if op == "gt": return a > b
        if op == "ge": return a >= b
    except (TypeError, ZeroDivisionError, ValueError, OverflowError): return None
    return None


def optimize_block(block: BasicBlock, *, const_prop=True, algebra=True, cse=False) -> dict[str, int]:
    """Apply optimizations that only need information from this basic block."""
    stats = {"constant_folds": 0, "algebraic_simplifications": 0, "cse_replacements": 0}
    constants: dict[str, Any] = {}
    rewritten: list[Instruction] = []

    def resolve(x):
        # Resolve only scalar variable operands; structured operands can contain
        # bound variables/formula references and must remain intact.
        return constants.get(x, x) if isinstance(x, str) else x

    for original in block.instructions:
        ins = Instruction(original.op, original.dst, list(original.args or []), original.label, original.line)

        # Any new assignment invalidates the old value first. Calls can modify
        # externally visible state, so do not carry assumptions across calls.
        if ins.op == "call":
            constants.clear()
        elif ins.op == "call_multi":
            constants.clear()
        if isinstance(ins.dst, str):
            constants.pop(ins.dst, None)
        elif isinstance(ins.dst, list):
            for dst in ins.dst:
                if isinstance(dst, str): constants.pop(dst, None)

        # Branch labels are not variable operands; only resolve the condition.
        if ins.op == "br" and len(ins.args) == 3:
            ins.args[0] = resolve(ins.args[0])
        elif ins.op not in ("jmp", "store"):
            ins.args = [resolve(a) for a in ins.args]

        if const_prop and ins.op in BINARY | {"neg", "not"}:
            folded = eval_op(ins.op, ins.args)
            if folded is not None:
                ins.op, ins.args = "const", [folded]
                stats["constant_folds"] += 1

        # Algebraic and boolean identities. These are safe for Funny's integer
        # and boolean values; do not apply them to structured operands.
        if algebra and ins.op in BINARY and len(ins.args) == 2:
            a, b = ins.args
            replacement = None
            has_replacement = True
            same_operand = type(a) is type(b) and a == b
            if ins.op == "eq" and same_operand: replacement = True
            elif ins.op == "ne" and same_operand: replacement = False
            elif ins.op == "and" and (a is False or b is False): replacement = False
            elif ins.op == "and" and a is True: replacement = b
            elif ins.op == "and" and b is True: replacement = a
            elif ins.op == "or" and (a is True or b is True): replacement = True
            elif ins.op == "or" and a is False: replacement = b
            elif ins.op == "or" and b is False: replacement = a
            elif ins.op == "implies" and a is False: replacement = True
            elif ins.op == "implies" and a is True: replacement = b
            elif ins.op == "implies" and b is True: replacement = True
            elif ins.op == "implies" and b is False:
                # a -> false is equivalent to not a
                ins.op, ins.args = "not", [a]
                stats["algebraic_simplifications"] += 1
                has_replacement = False
            elif ins.op == "add" and b == 0: replacement = a
            elif ins.op == "add" and a == 0: replacement = b
            elif ins.op == "sub" and b == 0: replacement = a
            elif ins.op == "mul" and b == 1: replacement = a
            elif ins.op == "mul" and a == 1: replacement = b
            elif ins.op == "mul" and (a == 0 or b == 0): replacement = 0
            elif ins.op == "div" and b == 1: replacement = a
            else: has_replacement = False
            if has_replacement and replacement is not None:
                ins.op, ins.args = "mov", [replacement]
                stats["algebraic_simplifications"] += 1

        # Track constants for both temporary and source variables, but only
        # inside this basic block. Since blocks end at control-flow boundaries,
        # this cannot leak a value across branches or loop iterations.
        if const_prop and isinstance(ins.dst, str):
            if ins.op == "const" and len(ins.args) == 1 and not isinstance(ins.args[0], (dict, list)):
                constants[ins.dst] = ins.args[0]
            elif ins.op == "mov" and len(ins.args) == 1 and not is_var(ins.args[0]) and not isinstance(ins.args[0], (dict, list)):
                constants[ins.dst] = ins.args[0]
            elif ins.op in BINARY | {"neg", "not"}:
                # The expression may have become a literal after simplification.
                folded = eval_op(ins.op, ins.args)
                if folded is not None:
                    ins.op, ins.args = "const", [folded]
                    constants[ins.dst] = folded
                    stats["constant_folds"] += 1
                else:
                    constants.pop(ins.dst, None)

        # A constant branch becomes an unconditional jump. This enables CFG
        # reachability pruning without trying to infer arbitrary path conditions.
        if ins.op == "br" and len(ins.args) == 3 and isinstance(ins.args[0], bool):
            target = ins.args[1] if ins.args[0] else ins.args[2]
            ins.op, ins.args = "jmp", [target]
            stats["algebraic_simplifications"] += 1

        rewritten.append(ins)

    if cse:
        available: dict[tuple, str] = {}
        for ins in rewritten:
            if ins.op in BINARY and ins.dst:
                key_args = tuple(ins.args)
                if ins.op in {"add", "mul", "eq", "ne", "and", "or"} and len(key_args) == 2:
                    key_args = tuple(sorted(key_args, key=repr))
                key = (ins.op, key_args)
                if key in available:
                    ins.op, ins.args = "mov", [available[key]]
                    stats["cse_replacements"] += 1
                else: available[key] = ins.dst
            elif ins.op in ("call", "call_multi", "store"):
                available.clear()

    block.instructions = rewritten
    return stats


def uses_of(ins: Instruction) -> set[str]:
    args = ins.args or []
    values = []
    if ins.op == "call": values = args[1:]
    elif ins.op == "call_multi": values = args[1:]
    elif ins.op == "store": values = args[1:]
    elif ins.op in ("jmp",): values = []
    elif ins.op == "br": values = [args[0]]
    elif ins.op == "return": values = args
    elif ins.op in ("assert", "assume"): values = args
    else: values = args
    result = set()
    def visit(x):
        if isinstance(x, str) and is_var(x): result.add(x)
        elif isinstance(x, dict):
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    for v in values: visit(v)
    return result


def _refresh_cfg_edges(blocks: list[BasicBlock]) -> None:
    """Recompute predecessor/successor lists after branch rewriting or pruning."""
    by_name = {b.name: b for b in blocks}
    for b in blocks:
        b.successors = []
        b.predecessors = []
    for index, block in enumerate(blocks):
        last = block.instructions[-1] if block.instructions else None
        if last and last.op == "jmp": targets = [last.args[0]]
        elif last and last.op == "br": targets = [last.args[1], last.args[2]]
        elif last and last.op == "return": targets = []
        elif index + 1 < len(blocks): targets = [blocks[index + 1].name]
        else: targets = []
        block.successors = list(dict.fromkeys(t for t in targets if t in by_name))
    for block in blocks:
        for successor in block.successors:
            by_name[successor].predecessors.append(block.name)


def _remove_unreachable_blocks(blocks: list[BasicBlock]) -> int:
    if not blocks: return 0
    _refresh_cfg_edges(blocks)
    by_name = {b.name: b for b in blocks}
    reachable, stack = set(), [blocks[0].name]
    while stack:
        name = stack.pop()
        if name in reachable or name not in by_name: continue
        reachable.add(name)
        stack.extend(by_name[name].successors)
    removed = len(blocks) - len(reachable)
    blocks[:] = [b for b in blocks if b.name in reachable]
    _refresh_cfg_edges(blocks)
    return removed


def _eliminate_dead_code(blocks: list[BasicBlock]) -> int:
    """Remove dead pure instructions using liveness across the whole CFG."""
    if not blocks:
        return 0

    by_name = {block.name: block for block in blocks}
    use: dict[str, set[str]] = {}
    defs: dict[str, set[str]] = {}

    for block in blocks:
        used, defined = set(), set()
        for ins in block.instructions:
            used.update(name for name in uses_of(ins) if name not in defined)
            if isinstance(ins.dst, str):
                defined.add(ins.dst)
            elif isinstance(ins.dst, list):
                defined.update(dst for dst in ins.dst if isinstance(dst, str))
        use[block.name] = used
        defs[block.name] = defined

    live_in = {block.name: set() for block in blocks}
    live_out = {block.name: set() for block in blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(blocks):
            out = set().union(*(
                live_in[successor]
                for successor in block.successors
                if successor in by_name
            ))
            inn = use[block.name] | (out - defs[block.name])
            if out != live_out[block.name] or inn != live_in[block.name]:
                live_out[block.name] = out
                live_in[block.name] = inn
                changed = True

    removed = 0
    for block in blocks:
        live = set(live_out[block.name])
        kept = []
        for ins in reversed(block.instructions):
            uses = uses_of(ins)
            dsts = [ins.dst] if isinstance(ins.dst, str) else (ins.dst or [])
            has_side_effect = ins.op not in PURE_OPS

            if dsts and all(dst not in live for dst in dsts) and not has_side_effect:
                removed += 1
                continue

            live.difference_update(dst for dst in dsts if isinstance(dst, str))
            live.update(uses)
            kept.append(ins)
        block.instructions = list(reversed(kept))

    return removed


def _optimization_shape(blocks: list[BasicBlock]) -> tuple:
    """Return the parts of the CFG that optimization is allowed to change."""
    return tuple(
        (block.name, tuple(
            (ins.op, ins.dst, repr(ins.args), ins.label)
            for ins in block.instructions
        ))
        for block in blocks
    )


def optimize_cfg(blocks: list[BasicBlock], **options) -> dict[str, int]:
    """Optimize blocks and CFG until stable, with optional global dead-code elimination."""
    totals = {
        "constant_folds": 0,
        "algebraic_simplifications": 0,
        "dead_instructions": 0,
        "cse_replacements": 0,
        "unreachable_blocks": 0,
    }
    options = dict(options)
    do_dce = options.pop("dce", True)

    while blocks:
        before = _optimization_shape(blocks)

        for block in blocks:
            stats = optimize_block(block, **options)
            for key, value in stats.items():
                totals[key] += value

        totals["unreachable_blocks"] += _remove_unreachable_blocks(blocks)
        if do_dce:
            totals["dead_instructions"] += _eliminate_dead_code(blocks)
            _refresh_cfg_edges(blocks)

        if _optimization_shape(blocks) == before:
            break

    return totals


# ---------------------------------------------------------------------------
# Liveness, interference graph and greedy coloring
# ---------------------------------------------------------------------------

def allocate_registers(blocks: list[BasicBlock]) -> dict[str, Any]:
    use, defs = {}, {}
    for b in blocks:
        u, d = set(), set()
        for ins in b.instructions:
            for name in uses_of(ins):
                if name not in d: u.add(name)
            if isinstance(ins.dst, str): d.add(ins.dst)
            elif isinstance(ins.dst, list): d.update(ins.dst)
        use[b.name], defs[b.name] = u, d
    live_in = {b.name: set() for b in blocks}; live_out = {b.name: set() for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            out = set().union(*(live_in[s] for s in b.successors)) if b.successors else set()
            inn = use[b.name] | (out - defs[b.name])
            if out != live_out[b.name] or inn != live_in[b.name]:
                live_out[b.name], live_in[b.name] = out, inn; changed = True
    graph: dict[str, set[str]] = {}
    for b in blocks:
        live = set(live_out[b.name])
        for ins in reversed(b.instructions):
            dsts = [ins.dst] if isinstance(ins.dst, str) else (ins.dst or [])
            for dst in dsts:
                if not isinstance(dst, str): continue
                graph.setdefault(dst, set())
                for other in live:
                    if other != dst:
                        graph.setdefault(other, set()).add(dst); graph[dst].add(other)
            live.difference_update(dsts)
            live.update(uses_of(ins))
    # Deterministic greedy coloring: highest degree first, then name.
    order = sorted(graph, key=lambda n: (-len(graph[n]), n))
    colors: dict[str, int] = {}
    for node in order:
        used = {colors[n] for n in graph[node] if n in colors}
        color = 0
        while color in used: color += 1
        colors[node] = color
    return {"live_in": {k: sorted(v) for k, v in live_in.items()},
            "live_out": {k: sorted(v) for k, v in live_out.items()},
            "interference": {k: sorted(v) for k, v in sorted(graph.items())},
            "colors": dict(sorted(colors.items())), "color_count": len(set(colors.values()))}


def interference_to_dot(graph: dict[str, list[str]]) -> str:
    lines = ["graph Interference {", "  node [shape=circle];"]
    edges = set()
    for node, neighbors in graph.items():
        lines.append(f'  "{node}";')
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))
            if edge not in edges:
                edges.add(edge); lines.append(f'  "{edge[0]}" -- "{edge[1]}";')
    lines.append("}")
    return "\n".join(lines)


def ir_text(functions, cfgs=None, title="IR"):
    lines = [f"=== {title} ==="]
    for fn in functions:
        lines.append(f"\nfunction {fn['name']}({', '.join(fn['parameters'])}) returns ({', '.join(fn['returns'])}):")
        for ins in fn["instructions"]:
            lines.append("  " + str(Instruction(**ins)))
        if cfgs and fn["name"] in cfgs:
            lines.append("  CFG blocks:")
            for b in cfgs[fn["name"]]:
                if isinstance(b, dict):
                    lines.append(f"    {b['name']}: succ={b['successors']}, pred={b['predecessors']}")
                else:
                    lines.append(f"    {b.name}: succ={b.successors}, pred={b.predecessors}")
    return "\n".join(lines)


def run_source(source: str, options: dict[str, bool]) -> dict[str, Any]:
    _, _, dfa = build_lexer(TOKEN_SPECS)
    tokens = dfa.tokenize(source)
    program = ast.Parser(tokens).parse()
    semantic = SemanticAnalyzer().analyze(program)
    if semantic["status"] != "PASS":
        return {"status": "SEMANTIC_ERROR", "semantic_errors": semantic["errors"]}
    generator = IRGenerator()
    original = generator.generate(program)
    optimized = json.loads(json.dumps(original))
    cfg_before, cfg_after, allocations, stats = {}, {}, {}, {}
    for fn in original["functions"]:
        cfg_before[fn["name"]] = build_cfg(fn)
    for fn in optimized["functions"]:
        blocks = build_cfg(fn)
        stats[fn["name"]] = optimize_cfg(blocks, **options)
        # Flatten optimized blocks back into linear instruction order.
        fn["instructions"] = [i.to_dict() for b in blocks for i in b.instructions]
        cfg_after[fn["name"]] = blocks
        allocations[fn["name"]] = allocate_registers(blocks)
    return {"status": "PASS", "semantic_analysis": semantic, "ir_before": original,
            "ir_after": optimized,
            "cfg_before": {n: [b.to_dict() for b in bs] for n, bs in cfg_before.items()},
            "cfg_after": {n: [b.to_dict() for b in bs] for n, bs in cfg_after.items()},
            "optimization_stats": stats, "register_allocation": allocations,
            "metrics": {"instructions_before": {f["name"]: len(f["instructions"]) for f in original["functions"]},
                        "instructions_after": {f["name"]: len(f["instructions"]) for f in optimized["functions"]},
                        "temporaries_before": {f["name"]: len({i.get("dst") for i in f["instructions"] if isinstance(i.get("dst"), str) and i["dst"].startswith("%t")}) for f in original["functions"]},
                        "temporaries_after": {f["name"]: len({i.get("dst") for i in f["instructions"] if isinstance(i.get("dst"), str) and i["dst"].startswith("%t")}) for f in optimized["functions"]},
                        "colors": {n: x["color_count"] for n, x in allocations.items()}}}


def run_self_tests() -> int:
    """Run compact built-in regression tests when no source file is supplied."""
    tests = [
        {
            "name": "arithmetic and constant propagation",
            "source": """main() returns r:int {
  a = 2 + 3;
  b = a * 1;
  r = b + 0;
}""",
            "check": lambda result: (
                result["metrics"]["instructions_after"]["main"] < result["metrics"]["instructions_before"]["main"]
                and result["optimization_stats"]["main"]["constant_folds"] >= 1
            ),
        },
        {
            "name": "if/else and CFG edges",
            "source": """main(x:int) returns r:int {
  if (x > 0) { r = 1; } else { r = 2; }
}""",
            "check": lambda result: (
                len(result["cfg_before"]["main"]) >= 4
                and any(block["successors"] for block in result["cfg_before"]["main"])
            ),
        },
        {
            "name": "while loop and back edge",
            "source": """main(n:int) returns r:int {
  i = 0;
  while (i < n) { i = i + 1; }
  r = i;
}""",
            "check": lambda result: any(
                successor == block["name"] or successor.startswith("while_cond_")
                for block in result["cfg_before"]["main"]
                for successor in block["successors"]
            ),
        },
        {
            "name": "CSE option",
            "source": """main(x:int) returns r:int {
  a = x + 2;
  b = x + 2;
  r = a + b;
}""",
            "options": {"const_prop": True, "algebra": True, "dce": True, "cse": True},
            "check": lambda result: result["optimization_stats"]["main"]["cse_replacements"] >= 1,
        },
        {
            "name": "register allocation metrics",
            "source": """main(x:int) returns r:int {
  a = x + 1;
  b = x + 2;
  r = a + b;
}""",
            "check": lambda result: (
                isinstance(result["register_allocation"]["main"]["colors"], dict)
                and result["metrics"]["colors"]["main"] >= 1
            ),
        },
        {
            "name": "constant propagation through source variables",
            "source": """main() returns r:int {
  b = 5;
  t = 0 + b;
  u = t - 5;
  r = u;
}""",
            "check": lambda result: (
                any(ins["op"] == "return" and ins["args"] == [0]
                    for ins in result["ir_after"]["functions"][0]["instructions"])
                and not any(ins.get("dst") in {"b", "t", "u", "r"}
                    for ins in result["ir_after"]["functions"][0]["instructions"])
            ),
        },
        {
            "name": "boolean simplification and unreachable CFG blocks",
            "source": """main(x:int) returns r:int {
  if (x == x) { r = 1; } else { r = 2; }
}""",
            "check": lambda result: (
                not any(block["name"].startswith("if_else_")
                    for block in result["cfg_after"]["main"])
                and not any(ins["op"] == "br"
                    for ins in result["ir_after"]["functions"][0]["instructions"])
                and any(ins["op"] == "return" and ins["args"] == ["r"]
                    for ins in result["ir_after"]["functions"][0]["instructions"])
            ),
        },
        {
            "name": "loop invariant and loop-carried variables",
            "source": """sum(n:int) returns r:int {
  i = 0;
  r = 0;
  while (i < n)
    invariant 0 <= i and r == i * (i - 1) / 2
  {
    r = r + i + 0;
    i = i + 1;
  }
}""",
            "check": lambda result: (
                {"le", "eq", "lt"}.issubset({
                    ins["op"] for ins in result["ir_after"]["functions"][0]["instructions"]
                })
                and {"i", "r"}.issubset({
                    ins.get("dst") for ins in result["ir_after"]["functions"][0]["instructions"]
                })
                and any(
                    ins["op"] == "jmp" and ins["args"][0].startswith("while_cond_")
                    for ins in result["ir_after"]["functions"][0]["instructions"]
                )
            ),
        },
    ]
    passed = 0
    print("Funny HW4 — встроенные тесты")
    print("=" * 38)
    for test in tests:
        options = test.get("options", {"const_prop": True, "algebra": True, "dce": True, "cse": False})
        try:
            result = run_source(test["source"], options)
            if result.get("status") != "PASS":
                raise AssertionError(f"pipeline status: {result.get('status')}; {result.get('semantic_errors', result.get('error'))}")
            if not test["check"](result):
                raise AssertionError("проверка результата не пройдена")
            passed += 1
            print(f"[PASS] {test['name']}")
        except Exception as exc:
            print(f"[FAIL] {test['name']}: {type(exc).__name__}: {exc}")
    print("-" * 38)
    print(f"Результат: {passed}/{len(tests)} тестов пройдено")
    return 0 if passed == len(tests) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Funny HW4: IR, CFG, optimizations and register allocation")
    parser.add_argument("source", nargs="?", help="path to a .funny source file; omit it to run built-in tests")
    parser.add_argument("--json", dest="json_path", default="hw4_output.json", help="JSON output path")
    parser.add_argument("--text", dest="text_path", default="hw4_output.txt", help="text output path")
    parser.add_argument("--dot-dir", default=None, help="directory for CFG/interference DOT files")
    parser.add_argument("--no-const-prop", action="store_true", help="disable constant propagation/folding")
    parser.add_argument("--no-algebra", action="store_true", help="disable algebraic simplification")
    parser.add_argument("--no-dce", action="store_true", help="disable dead-code elimination")
    parser.add_argument("--cse", action="store_true", help="enable common subexpression elimination")
    args = parser.parse_args()
    if not args.source:
        return run_self_tests()
    source = Path(args.source).read_text(encoding="utf-8")
    options = {"const_prop": not args.no_const_prop, "algebra": not args.no_algebra,
               "dce": not args.no_dce, "cse": args.cse}
    try:
        result = run_source(source, options)
    except Exception as exc:
        result = {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}
    Path(args.json_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.get("status") == "PASS":
        text = ir_text(result["ir_before"]["functions"], title="IR BEFORE OPTIMIZATION") + "\n\n"
        text += ir_text(result["ir_after"]["functions"], result.get("cfg_after"), title="IR AFTER OPTIMIZATION")
        text += "\n\n=== METRICS ===\n" + json.dumps(result["metrics"], ensure_ascii=False, indent=2)
        text += "\n\n=== OPTIMIZATION STATS ===\n" + json.dumps(result["optimization_stats"], ensure_ascii=False, indent=2)
        for name, alloc in result["register_allocation"].items():
            text += f"\n\n=== REGISTER ALLOCATION: {name} ===\n" + json.dumps(alloc, ensure_ascii=False, indent=2)
        Path(args.text_path).write_text(text, encoding="utf-8")
        if args.dot_dir:
            outdir = Path(args.dot_dir); outdir.mkdir(parents=True, exist_ok=True)
            for name, blocks in result["cfg_after"].items():
                # Rebuild typed blocks from the JSON representation for stable DOT output.
                typed = [BasicBlock(b["name"], [Instruction(**i) for i in b["instructions"]], b["successors"], b["predecessors"]) for b in blocks]
                (outdir / f"cfg_{name}.dot").write_text(cfg_to_dot(name, typed), encoding="utf-8")
                graph = result["register_allocation"][name]["interference"]
                (outdir / f"interference_{name}.dot").write_text(interference_to_dot(graph), encoding="utf-8")
        print(f"Status: {result['status']}")
        print(f"JSON: {args.json_path}")
        print(f"Text: {args.text_path}")
        print("Instructions before:", result["metrics"]["instructions_before"])
        print("Instructions after:", result["metrics"]["instructions_after"])
        print("Colors:", result["metrics"]["colors"])
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
