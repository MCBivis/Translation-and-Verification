from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import funny_parser as hw2
from funny_lexer import TOKEN_SPECS, build_lexer

INT = "int"
BOOL = "bool"
UNKNOWN = "<unknown>"


class SemanticDiagnostic:
    def __init__(self, message: str, node: Any = None, code: str = "SEMANTIC_ERROR"):
        self.message = message
        self.code = code
        self.line = getattr(node, "line", None) if node is not None else None
        self.column = getattr(node, "column", None) if node is not None else None

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.line is not None and self.column is not None:
            result["line"] = self.line
            result["column"] = self.column
        else:
            result["position"] = "unknown"
        return result

    def __str__(self) -> str:
        pos = f"{self.line}:{self.column}: " if self.line is not None and self.column is not None else ""
        return f"{pos}{self.code}: {self.message}"


class Symbol:
    def __init__(self, name: str, type_name: str | None, kind: str, node: Any = None):
        self.name = name
        self.type = type_name
        self.kind = kind
        self.node = node

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type or "inferred", "kind": self.kind}


class SemanticAnalyzer:
    def __init__(self):
        self.errors: list[SemanticDiagnostic] = []
        self.functions: dict[str, Any] = {}
        self.scopes: list[dict[str, Symbol]] = []
        self.current_function: Any = None
        self.inferred_symbols: list[Symbol] = []
        self.expression_types: dict[int, str] = {}
        self.annotations: list[dict[str, Any]] = []

    def error(self, message: str, node: Any = None, code: str = "SEMANTIC_ERROR") -> None:
        self.errors.append(SemanticDiagnostic(message, node, code))

    @staticmethod
    def type_name(type_node: Any) -> str:
        if type_node is None:
            return UNKNOWN
        name = getattr(type_node, "name", str(type_node))
        is_array = bool(getattr(type_node, "is_array", False))
        return f"{name}[]" if is_array else name

    def analyze(self, program: Any) -> dict[str, Any]:
        self.errors = []
        self.functions = {}
        self.inferred_symbols = []
        self.expression_types = {}
        self.annotations = []

        # Pass 1: collect all function signatures, allowing calls to later functions.
        for fn in program.functions:
            if fn.name in self.functions:
                self.error(f"function '{fn.name}' is declared more than once", fn, "DUPLICATE_FUNCTION")
            else:
                self.functions[fn.name] = fn

        # Pass 2: validate function declarations and bodies.
        for fn in program.functions:
            self.check_function(fn)

        for symbol in self.inferred_symbols:
            if symbol.type is None:
                self.error(f"cannot infer a type for local variable '{symbol.name}'", symbol.node, "TYPE_INFERENCE_FAILED")

        return {
            "ast": self.ast_to_dict(program),
            "function_signatures": {
                name: {
                    "parameters": [self.type_name(v.type) for v in fn.parameters],
                    "returns": [self.type_name(v.type) for v in fn.returns],
                }
                for name, fn in self.functions.items()
            },
            "inferred_locals": [s.to_dict() for s in self.inferred_symbols],
            "expression_types": self.annotations,
            "errors": [e.to_dict() for e in self.errors],
            "status": "PASS" if not self.errors else "FAIL",
        }

    def push_scope(self) -> None:
        self.scopes.append({})

    def pop_scope(self) -> None:
        self.scopes.pop()

    def declare(self, name: str, type_name: str | None, kind: str, node: Any = None) -> Symbol:
        scope = self.scopes[-1]
        if name in scope:
            self.error(f"'{name}' is declared more than once in the same scope", node, "DUPLICATE_NAME")
            return scope[name]
        symbol = Symbol(name, type_name, kind, node)
        scope[name] = symbol
        return symbol

    def lookup(self, name: str) -> Symbol | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def check_function(self, fn: Any) -> None:
        self.current_function = fn
        self.push_scope()
        # Funny specification: requires can see parameters only.
        for param in fn.parameters:
            self.declare(param.name, self.type_name(param.type), "parameter", param)
        self.check_predicate_if_present(getattr(fn, "requires", None))

        # ensures can see parameters and return variables.
        for ret in fn.returns:
            self.declare(ret.name, self.type_name(ret.type), "return", ret)
        self.check_predicate_if_present(getattr(fn, "ensures", None))

        # Uses and inferred locals belong to the function scope. Names of parameters,
        # returns and locals must be unique throughout a function (Funny specification).
        for local in getattr(fn, "uses", []):
            local_type = self.type_name(local.type) if getattr(local, "type", None) is not None else None
            symbol = self.declare(local.name, local_type, "local", local)
            if local_type is None and symbol not in self.inferred_symbols:
                self.inferred_symbols.append(symbol)
        self.check_statement(fn.body)
        self.pop_scope()
        self.current_function = None

    def check_predicate_if_present(self, predicate: Any) -> None:
        if predicate is not None:
            actual = self.check_predicate(predicate)
            if actual != BOOL and actual != UNKNOWN:
                self.error(f"condition must have type bool, got '{actual}'", predicate, "TYPE_MISMATCH")

    def check_statement(self, stmt: Any) -> None:
        if stmt is None:
            return
        kind = type(stmt).__name__
        if kind == "Block":
            # Blocks do not introduce new variable declarations in this Funny subset;
            # inferred locals stay visible throughout the function.
            for child in stmt.statements:
                self.check_statement(child)
        elif kind == "Assignment":
            self.check_assignment(stmt)
        elif kind == "IfStatement":
            self.check_predicate_if_present(stmt.condition)
            self.check_statement(stmt.then_branch)
            self.check_statement(getattr(stmt, "else_branch", None))
        elif kind == "WhileStatement":
            self.check_predicate_if_present(stmt.condition)
            self.check_predicate_if_present(getattr(stmt, "invariant", None))
            self.check_statement(stmt.body)
        elif kind in ("AssertStatement", "AssumeStatement"):
            self.check_predicate_if_present(stmt.predicate)
        else:
            self.error(f"unsupported statement node '{kind}'", stmt, "UNSUPPORTED_NODE")

    def check_assignment(self, stmt: Any) -> None:
        targets = list(stmt.targets)
        value = stmt.value
        # A call may return multiple values, but only a tuple assignment can receive them.
        if type(value).__name__ == "FunctionCall":
            if value.name == "length":
                value_type = self.check_expression(value)
                if len(targets) != 1:
                    self.error("single-value function call must be assigned to exactly one target", stmt, "RETURN_COUNT_MISMATCH")
                    return
                self.assign_target(targets[0], value_type, stmt)
                return
            fn = self.functions.get(value.name)
            self.check_call_arguments(value, fn)
            if fn is None:
                self.error(f"unknown function '{value.name}'", value, "UNKNOWN_FUNCTION")
                return
            return_types = [self.type_name(v.type) for v in fn.returns]
            if len(return_types) != len(targets):
                self.error(
                    f"function '{value.name}' returns {len(return_types)} value(s), but assignment has {len(targets)} target(s)",
                    stmt, "RETURN_COUNT_MISMATCH"
                )
                return
            for target, actual_type in zip(targets, return_types):
                self.assign_target(target, actual_type, stmt)
            return

        value_type = self.check_expression(value)
        if len(targets) != 1:
            self.error("multiple assignment requires a function call returning the same number of values", stmt, "RETURN_COUNT_MISMATCH")
            return
        self.assign_target(targets[0], value_type, stmt)

    def assign_target(self, target: Any, value_type: str, node: Any) -> None:
        kind = type(target).__name__
        if kind == "Variable":
            name = target.name
            symbol = self.lookup(name)
            if symbol is None:
                # Bonus: first assignment introduces an inferred local variable.
                if value_type == BOOL:
                    self.error("Funny has no bool variable type; booleans are only used in conditions and predicates", target, "UNSUPPORTED_VARIABLE_TYPE")
                    inferred_type = None
                else:
                    inferred_type = value_type if value_type != UNKNOWN else None
                symbol = self.declare(name, inferred_type, "inferred_local", target)
                self.inferred_symbols.append(symbol)
                self.annotations.append({"variable": name, "inferred_type": symbol.type or "unresolved"})
            elif symbol.kind == "parameter":
                self.error(f"parameter '{name}' is read-only and cannot be assigned", target, "ASSIGN_TO_PARAMETER")
            elif symbol.type is None:
                if value_type != UNKNOWN:
                    if value_type == BOOL:
                        self.error("Funny has no bool variable type; booleans are only used in conditions and predicates", target, "UNSUPPORTED_VARIABLE_TYPE")
                    else:
                        symbol.type = value_type
                        if symbol.kind == "local" and symbol not in self.inferred_symbols:
                            self.inferred_symbols.append(symbol)
                        self.annotations.append({"variable": name, "inferred_type": value_type})
            elif value_type != UNKNOWN and symbol.type != value_type:
                self.error(f"cannot assign '{value_type}' to '{name}' of type '{symbol.type}'", target, "TYPE_MISMATCH")
            try:
                target.inferred_type = symbol.type or UNKNOWN
            except Exception:
                pass
        elif kind == "ArrayAccess":
            array_name = getattr(target, "array", None)
            symbol = self.lookup(array_name) if array_name else None
            if symbol is None:
                # Funny defines a[i] = value as functional array update; this
                # also lets the first assignment infer a local array's type.
                symbol = self.declare(array_name, "int[]", "inferred_local", target)
                self.inferred_symbols.append(symbol)
                self.annotations.append({"variable": array_name, "inferred_type": "int[]"})
            elif symbol.kind == "parameter":
                self.error(f"parameter array '{array_name}' is read-only and cannot be updated", target, "ASSIGN_TO_PARAMETER")
            elif symbol.type is None:
                symbol.type = "int[]"
                if symbol not in self.inferred_symbols:
                    self.inferred_symbols.append(symbol)
                self.annotations.append({"variable": array_name, "inferred_type": "int[]"})
            if symbol.type not in ("int[]", UNKNOWN):
                self.error(f"'{array_name}' is not an array (type '{symbol.type}')", target, "TYPE_MISMATCH")
            index_type = self.check_expression(target.index)
            if index_type not in (INT, UNKNOWN):
                self.error(f"array index must have type int, got '{index_type}'", target, "TYPE_MISMATCH")
            if value_type not in (INT, UNKNOWN):
                self.error(f"array element must have type int, got '{value_type}'", target, "TYPE_MISMATCH")
            try:
                target.inferred_type = INT
            except Exception:
                pass
        else:
            self.error(f"invalid assignment target '{kind}'", target, "INVALID_ASSIGNMENT_TARGET")

    def check_expression(self, expr: Any) -> str:
        if expr is None:
            return UNKNOWN
        kind = type(expr).__name__
        result = UNKNOWN
        if kind == "IntLiteral":
            result = INT
        elif kind == "BoolLiteral":
            result = BOOL
        elif kind == "Variable":
            symbol = self.lookup(expr.name)
            if symbol is None:
                self.error(f"unknown name '{expr.name}'", expr, "UNKNOWN_NAME")
            elif symbol.type is None:
                self.error(f"type of local variable '{expr.name}' is not known yet; assign it before use", expr, "USE_BEFORE_INFERENCE")
            else:
                result = symbol.type
        elif kind == "ArrayAccess":
            symbol = self.lookup(expr.array)
            if symbol is None:
                self.error(f"unknown array '{expr.array}'", expr, "UNKNOWN_NAME")
            elif symbol.type not in ("int[]", UNKNOWN):
                self.error(f"'{expr.array}' is not an array (type '{symbol.type}')", expr, "TYPE_MISMATCH")
            index_type = self.check_expression(expr.index)
            if index_type not in (INT, UNKNOWN):
                self.error(f"array index must have type int, got '{index_type}'", expr, "TYPE_MISMATCH")
            result = INT
        elif kind == "FunctionCall":
            if expr.name == "length":
                if len(expr.arguments) != 1:
                    self.error(f"built-in 'length' expects 1 argument, got {len(expr.arguments)}", expr, "ARGUMENT_COUNT_MISMATCH")
                for arg in expr.arguments:
                    arg_type = self.check_expression(arg)
                    if arg_type not in ("int[]", UNKNOWN):
                        self.error(f"argument of 'length' must have type 'int[]', got '{arg_type}'", expr, "ARGUMENT_TYPE_MISMATCH")
                result = INT
            else:
                fn = self.functions.get(expr.name)
                self.check_call_arguments(expr, fn)
                if fn is None:
                    self.error(f"unknown function '{expr.name}'", expr, "UNKNOWN_FUNCTION")
                elif len(fn.returns) == 1:
                    result = self.type_name(fn.returns[0].type)
                elif len(fn.returns) > 1:
                    self.error(f"function '{expr.name}' returns multiple values and cannot be used as a single expression", expr, "MULTIPLE_RETURN_IN_EXPRESSION")
        elif kind == "UnaryOp":
            operand = self.check_expression(expr.operand)
            if expr.operator == "-":
                if operand not in (INT, UNKNOWN):
                    self.error(f"unary '-' requires int, got '{operand}'", expr, "TYPE_MISMATCH")
                result = INT
            elif expr.operator == "not":
                if operand not in (BOOL, UNKNOWN):
                    self.error(f"'not' requires bool, got '{operand}'", expr, "TYPE_MISMATCH")
                result = BOOL
        elif kind == "BinaryOp":
            left = self.check_expression(expr.left)
            right = self.check_expression(expr.right)
            if expr.operator in ("+", "-", "*", "/"):
                self.require_type(left, INT, expr, f"left operand of '{expr.operator}'")
                self.require_type(right, INT, expr, f"right operand of '{expr.operator}'")
                result = INT
            elif expr.operator in ("and", "or", "->"):
                self.require_type(left, BOOL, expr, f"left operand of '{expr.operator}'")
                self.require_type(right, BOOL, expr, f"right operand of '{expr.operator}'")
                result = BOOL
        else:
            self.error(f"unsupported expression node '{kind}'", expr, "UNSUPPORTED_NODE")
        self.expression_types[id(expr)] = result
        try:
            expr.inferred_type = result
        except Exception:
            pass
        self.annotations.append({"node": kind, "type": result, "value": self.node_preview(expr)})
        return result

    def check_predicate(self, pred: Any) -> str:
        if pred is None:
            return BOOL
        kind = type(pred).__name__
        if kind in ("BoolLiteral", "PredicateNot", "PredicateBinary", "Comparison", "Quantifier", "FormulaRef"):
            if kind == "BoolLiteral":
                result = BOOL
            elif kind == "PredicateNot":
                operand = self.check_predicate(pred.operand)
                self.require_type(operand, BOOL, pred, "operand of 'not'")
                result = BOOL
            elif kind == "PredicateBinary":
                left = self.check_predicate(pred.left)
                right = self.check_predicate(pred.right)
                self.require_type(left, BOOL, pred, f"left operand of '{pred.operator}'")
                self.require_type(right, BOOL, pred, f"right operand of '{pred.operator}'")
                result = BOOL
            elif kind == "Comparison":
                left = self.check_expression(pred.left)
                right = self.check_expression(pred.right)
                if left != UNKNOWN and right != UNKNOWN and left != right:
                    self.error(f"cannot compare '{left}' with '{right}'", pred, "TYPE_MISMATCH")
                if pred.operator in ("<", "<=", ">", ">=") and left not in (INT, UNKNOWN):
                    self.error(f"operator '{pred.operator}' requires integer operands", pred, "TYPE_MISMATCH")
                result = BOOL
            elif kind == "Quantifier":
                self.push_scope()
                variable = pred.variable
                self.declare(variable.name, self.type_name(variable.type), "quantified_variable", variable)
                body_type = self.check_predicate(pred.body)
                self.require_type(body_type, BOOL, pred, "quantifier body")
                self.pop_scope()
                result = BOOL
            else:  # FormulaRef
                for arg in pred.arguments:
                    self.check_expression(arg)
                # Formula declarations are outside the currently supported subset.
                self.error(f"formula reference '{pred.name}' cannot be resolved: formula declarations are not supported", pred, "UNKNOWN_FORMULA")
                result = BOOL
        else:
            # Some parser versions represent a predicate as an expression tree.
            result = self.check_expression(pred)
            if result != BOOL and result != UNKNOWN:
                self.error(f"predicate must be bool, got '{result}'", pred, "TYPE_MISMATCH")
                result = BOOL
        self.expression_types[id(pred)] = result
        try:
            pred.inferred_type = result
        except Exception:
            pass
        self.annotations.append({"node": kind, "type": result, "value": self.node_preview(pred)})
        return result

    def check_call_arguments(self, call: Any, fn: Any) -> None:
        arg_types = [self.check_expression(arg) for arg in call.arguments]
        if fn is None:
            return
        params = list(fn.parameters)
        if len(arg_types) != len(params):
            self.error(f"function '{call.name}' expects {len(params)} argument(s), got {len(arg_types)}", call, "ARGUMENT_COUNT_MISMATCH")
        for index, (actual, param) in enumerate(zip(arg_types, params), start=1):
            expected = self.type_name(param.type)
            if actual != UNKNOWN and actual != expected:
                self.error(f"argument {index} of '{call.name}' must have type '{expected}', got '{actual}'", call, "ARGUMENT_TYPE_MISMATCH")

    def require_type(self, actual: str, expected: str, node: Any, context: str) -> None:
        if actual not in (expected, UNKNOWN):
            self.error(f"{context} must have type '{expected}', got '{actual}'", node, "TYPE_MISMATCH")

    @staticmethod
    def node_preview(node: Any) -> str:
        kind = type(node).__name__
        if kind in ("Variable", "FunctionCall"):
            return getattr(node, "name", "")
        if kind == "IntLiteral":
            return str(node.value)
        if kind == "BoolLiteral":
            return str(node.value).lower()
        if kind in ("BinaryOp", "Comparison", "PredicateBinary"):
            return getattr(node, "operator", "")
        return kind

    @classmethod
    def ast_to_dict(cls, value: Any) -> Any:
        if is_dataclass(value):
            result = {"node": type(value).__name__}
            for f in fields(value):
                result[f.name] = cls.ast_to_dict(getattr(value, f.name))
            if hasattr(value, "inferred_type"):
                result["inferred_type"] = getattr(value, "inferred_type")
            return result
        if isinstance(value, list):
            return [cls.ast_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {str(k): cls.ast_to_dict(v) for k, v in value.items()}
        return value

def parse_source(source: str) -> Any:
    _, _, dfa = build_lexer(TOKEN_SPECS)
    tokens = dfa.tokenize(source)
    parser = hw2.Parser(tokens)
    program = parser.parse()
    # Accommodate parser versions that leave trailing tokens for the caller to check.
    if getattr(parser, "current", None) is not None:
        token = parser.current
        raise hw2.ParseError(f"unexpected token {token.type} ({token.lexeme!r})", token)
    return program


def run_semantic_tests() -> list[dict[str, Any]]:
    cases = [
        ("valid_arithmetic_and_call", """add(a:int, b:int) returns r:int { r = a + b; }\nmain() returns r:int { r = add(2, 3); }""", True, []),
        ("valid_inferred_locals", """split(x:int) returns a:int, b:int { a = x / 2; b = x - a; }\nmain() returns r:int { p, q = split(9); r = p + q; }""", True, []),
        ("unknown_variable", """main() returns r:int { r = missing + 1; }""", False, ["UNKNOWN_NAME"]),
        ("duplicate_parameter", """bad(a:int, a:int) returns r:int { r = a; }""", False, ["DUPLICATE_NAME"]),
        ("incompatible_assignment", """main(a:int[]) returns r:int { r = a; }""", False, ["TYPE_MISMATCH"]),
        ("wrong_argument_count", """one(a:int) returns r:int { r = a; }\nmain() returns r:int { r = one(); }""", False, ["ARGUMENT_COUNT_MISMATCH"]),
        ("wrong_argument_type", """one(a:int) returns r:int { r = a; }\nmain(a:int[]) returns r:int { r = one(a); }""", False, ["ARGUMENT_TYPE_MISMATCH"]),
        ("unknown_function", """main() returns r:int { r = nope(1); }""", False, ["UNKNOWN_FUNCTION"]),
        ("wrong_condition_operand_types", """main(a:int[]) returns r:int { if (1 == a) { r = 2; } }""", False, ["TYPE_MISMATCH"]),
        ("array_update_and_length", """main() returns r:int { a[0] = 7; r = length(a); }""", True, []),
        ("untyped_unused_local", """main() returns r:int uses temp { r = 1; }""", False, ["TYPE_INFERENCE_FAILED"]),
        ("typed_array_local", """main() returns r:int uses a:int[] { a[0] = 7; r = a[0]; }""", True, []),
    ]
    output = []
    for name, source, should_pass, expected_codes in cases:
        item: dict[str, Any] = {"name": name, "input": source, "expected": "PASS" if should_pass else "FAIL"}
        try:
            ast = parse_source(source)
            result = SemanticAnalyzer().analyze(ast)
            actual_codes = [e["code"] for e in result["errors"]]
            passed = (not result["errors"]) if should_pass else any(code in actual_codes for code in expected_codes)
            item.update({"actual": result["status"], "errors": result["errors"], "annotated_ast": result["ast"], "expression_types": result["expression_types"], "status": "PASS" if passed else "FAIL"})
        except Exception as exc:
            item.update({"actual": "PARSE_ERROR", "error_type": type(exc).__name__, "error": str(exc), "status": "FAIL"})
        output.append(item)
    return output


def main() -> int:
    cli = argparse.ArgumentParser(description="Funny HW3 semantic analysis")
    cli.add_argument("source", nargs="?", help="path to a .funny source file; if omitted, run tests")
    cli.add_argument("--report", default="semantic_test_results.json", help="JSON report path")
    args = cli.parse_args()

    if args.source:
        source = Path(args.source).read_text(encoding="utf-8")
        try:
            program = parse_source(source)
            result = SemanticAnalyzer().analyze(program)
        except Exception as exc:
            result = {"status": "FAIL", "errors": [{"code": "PARSE_ERROR", "message": str(exc)}]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "PASS" else 1

    tests = run_semantic_tests()
    passed = sum(test["status"] == "PASS" for test in tests)
    report = {"tests_total": len(tests), "tests_passed": passed, "tests_failed": len(tests) - passed,
              "status": "PASS" if passed == len(tests) else "FAIL", "tests": tests}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Semantic tests: {passed}/{len(tests)} PASS")
    for test in tests:
        print(f"[{test['status']}] {test['name']}")
    print(f"Test report: {args.report}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
