from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ============================================================
# AST
# ============================================================

@dataclass(kw_only=True)
class ASTNode:
    """Base class for AST nodes with their source position."""

    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class Program(ASTNode):
    functions: list["Function"]


@dataclass
class Function(ASTNode):
    name: str
    parameters: list["VariableDef"]
    returns: list["VariableDef"]
    requires: Optional["Predicate"]
    ensures: Optional["Predicate"]
    uses: list["LocalVarDef"]
    body: "Statement"


@dataclass
class VariableDef(ASTNode):
    name: str
    type: "Type"


@dataclass
class LocalVarDef(ASTNode):
    name: str
    type: Optional["Type"]


@dataclass
class Type(ASTNode):
    name: str
    is_array: bool = False


# ============================================================
# Statements
# ============================================================

class Statement(ASTNode):
    pass


@dataclass
class Block(Statement):
    statements: list[Statement]


@dataclass
class Assignment(Statement):
    targets: list["Expression"]
    value: "Expression"


@dataclass
class IfStatement(Statement):
    condition: "Predicate"
    then_branch: Statement
    else_branch: Optional[Statement]


@dataclass
class WhileStatement(Statement):
    condition: "Predicate"
    invariant: Optional["Predicate"]
    body: Statement


@dataclass
class AssertStatement(Statement):
    predicate: "Predicate"


@dataclass
class AssumeStatement(Statement):
    predicate: "Predicate"


# ============================================================
# Expressions
# ============================================================

class Expression(ASTNode):
    pass


@dataclass
class IntLiteral(Expression):
    value: int


@dataclass
class Variable(Expression):
    name: str


@dataclass
class ArrayAccess(Expression):
    array: str
    index: Expression


@dataclass
class FunctionCall(Expression):
    name: str
    arguments: list[Expression]


@dataclass
class UnaryOp(Expression):
    operator: str
    operand: Expression


@dataclass
class BinaryOp(Expression):
    operator: str
    left: Expression
    right: Expression


# ============================================================
# Predicates
# ============================================================

class Predicate(ASTNode):
    pass


@dataclass
class BoolLiteral(Predicate):
    value: bool


@dataclass
class Comparison(Predicate):
    operator: str
    left: Expression
    right: Expression


@dataclass
class PredicateNot(Predicate):
    operand: Predicate


@dataclass
class PredicateBinary(Predicate):
    operator: str
    left: Predicate
    right: Predicate


@dataclass
class Quantifier(Predicate):
    kind: str
    variable: VariableDef
    body: Predicate


@dataclass
class FormulaRef(Predicate):
    name: str
    arguments: list[Expression]


# ============================================================
# Parser error
# ============================================================

class ParseError(Exception):
    def __init__(self, message: str, token=None):
        if token is not None:
            message = (
                f"{message} "
                f"at {token.line}:{token.column}"
            )

        super().__init__(message)


# ============================================================
# Parser
# ============================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # --------------------------------------------------------
    # Basic token operations
    # --------------------------------------------------------

    @property
    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]

        return None

    def peek(self, token_type: str) -> bool:
        token = self.current
        return token is not None and token.type == token_type

    def advance(self):
        token = self.current

        if token is not None:
            self.pos += 1

        return token

    def expect(self, token_type: str):
        token = self.current

        if token is None:
            raise ParseError(
                f"expected {token_type}, got end of input"
            )

        if token.type != token_type:
            raise ParseError(
                f"expected {token_type}, got {token.lexeme!r}",
                token,
            )

        self.pos += 1
        return token

    def expect_identifier(self):
        token = self.current

        if token is None:
            raise ParseError(
                "expected identifier, got end of input"
            )

        if token.type != "IDENT":
            raise ParseError(
                f"expected identifier, got {token.lexeme!r}",
                token,
            )

        self.pos += 1
        return token

    # --------------------------------------------------------
    # Entry point
    # --------------------------------------------------------

    def parse(self) -> Program:
        if self.current is None:
            raise ParseError("empty input")

        start_token = self.current
        functions = []

        while self.current is not None:
            functions.append(self.parse_function())

        return Program(
            functions,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # function
    #
    # identifier "(" parameters? ")"
    # requires?
    # returns return_variables
    # ensures?
    # uses?
    # statement
    # --------------------------------------------------------

    def parse_function(self) -> Function:
        start_token = self.current
        name = self.expect_identifier().lexeme

        self.expect("LPAREN")

        parameters = []

        if not self.peek("RPAREN"):
            parameters = self.parse_parameters()

        self.expect("RPAREN")

        requires = None

        if self.peek("KW_REQUIRES"):
            self.advance()
            requires = self.parse_predicate()

        self.expect("KW_RETURNS")

        returns = self.parse_return_variables()

        ensures = None

        if self.peek("KW_ENSURES"):
            self.advance()
            ensures = self.parse_predicate()

        uses = []

        if self.peek("KW_USES"):
            self.advance()
            uses = self.parse_uses()

        body = self.parse_statement()

        return Function(
            name=name,
            parameters=parameters,
            returns=returns,
            requires=requires,
            ensures=ensures,
            uses=uses,
            body=body,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # parameters
    # --------------------------------------------------------

    def parse_parameters(self):
        result = [self.parse_variable_def()]

        while self.peek("COMMA"):
            self.advance()
            result.append(self.parse_variable_def())

        return result

    def parse_return_variables(self):
        result = [self.parse_variable_def()]

        while self.peek("COMMA"):
            self.advance()
            result.append(self.parse_variable_def())

        return result

    def parse_variable_def(self):
        start_token = self.current
        name = self.expect_identifier().lexeme

        self.expect("COLON")

        type_ = self.parse_type()

        return VariableDef(
            name,
            type_,
            line=start_token.line,
            column=start_token.column,
        )

    def parse_local_var_def(self):
        start_token = self.current
        name = self.expect_identifier().lexeme

        type_ = None

        if self.peek("COLON"):
            self.advance()
            type_ = self.parse_type()

        return LocalVarDef(
            name,
            type_,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # type
    # --------------------------------------------------------

    def parse_type(self):
        start_token = self.expect("KW_INT")

        if self.peek("LBRACKET"):
            self.advance()
            self.expect("RBRACKET")

            return Type(
                "int",
                is_array=True,
                line=start_token.line,
                column=start_token.column,
            )

        return Type(
            "int",
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # uses
    # --------------------------------------------------------

    def parse_uses(self):
        result = [self.parse_local_var_def()]

        while self.peek("COMMA"):
            self.advance()
            result.append(self.parse_local_var_def())

        return result

    # --------------------------------------------------------
    # statements
    # --------------------------------------------------------

    def parse_statement(self):
        if self.peek("LBRACE"):
            return self.parse_block()

        if self.peek("KW_IF"):
            return self.parse_if()

        if self.peek("KW_WHILE"):
            return self.parse_while()

        if self.peek("KW_ASSERT"):
            return self.parse_assert()

        if self.peek("KW_ASSUME"):
            return self.parse_assume()

        if self.peek("IDENT"):
            return self.parse_assignment()

        token = self.current

        if token is None:
            raise ParseError("expected statement, got end of input")

        raise ParseError(
            f"expected statement, got {token.lexeme!r}",
            token,
        )

    # --------------------------------------------------------
    # block
    # --------------------------------------------------------

    def parse_block(self):
        start_token = self.expect("LBRACE")

        statements = []

        while self.current is not None and not self.peek("RBRACE"):
            statements.append(self.parse_statement())

        self.expect("RBRACE")

        return Block(
            statements,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # if
    # --------------------------------------------------------

    def parse_if(self):
        start_token = self.expect("KW_IF")
        self.expect("LPAREN")

        condition = self.parse_condition()

        self.expect("RPAREN")

        then_branch = self.parse_statement()

        else_branch = None

        if self.peek("KW_ELSE"):
            self.advance()
            else_branch = self.parse_statement()

        return IfStatement(
            condition,
            then_branch,
            else_branch,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # while
    # --------------------------------------------------------

    def parse_while(self):
        start_token = self.expect("KW_WHILE")
        self.expect("LPAREN")

        condition = self.parse_condition()

        self.expect("RPAREN")

        invariant = None

        if self.peek("KW_INVARIANT"):
            self.advance()
            invariant = self.parse_predicate()

        body = self.parse_statement()

        return WhileStatement(
            condition,
            invariant,
            body,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # assert
    # --------------------------------------------------------

    def parse_assert(self):
        start_token = self.expect("KW_ASSERT")

        predicate = self.parse_predicate()

        self.expect("SEMICOLON")

        return AssertStatement(
            predicate,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # assume
    # --------------------------------------------------------

    def parse_assume(self):
        start_token = self.expect("KW_ASSUME")

        predicate = self.parse_predicate()

        self.expect("SEMICOLON")

        return AssumeStatement(
            predicate,
            line=start_token.line,
            column=start_token.column,
        )

    # --------------------------------------------------------
    # assignment
    # --------------------------------------------------------

    def parse_assignment(self):
        start_token = self.current
        first = self.expect_identifier().lexeme

        # ----------------------------------------------------
        # Tuple assignment:
        #
        # p, q = split(9);
        # ----------------------------------------------------

        if self.peek("COMMA"):
            targets = [Variable(
                first,
                line=start_token.line,
                column=start_token.column,
            )]

            while self.peek("COMMA"):
                self.advance()

                target_token = self.current
                name = self.expect_identifier().lexeme
                targets.append(Variable(
                    name,
                    line=target_token.line,
                    column=target_token.column,
                ))

            self.expect("ASSIGN")

            value = self.parse_function_call()

            self.expect("SEMICOLON")

            return Assignment(
                targets,
                value,
                line=start_token.line,
                column=start_token.column,
            )

        # ----------------------------------------------------
        # Single assignment.
        # ----------------------------------------------------

        target = Variable(
            first,
            line=start_token.line,
            column=start_token.column,
        )

        if self.peek("LBRACKET"):
            self.advance()

            index = self.parse_expression()

            self.expect("RBRACKET")

            target = ArrayAccess(
                first,
                index,
                line=start_token.line,
                column=start_token.column,
            )

        self.expect("ASSIGN")

        value = self.parse_expression()

        self.expect("SEMICOLON")

        return Assignment(
            [target],
            value,
            line=start_token.line,
            column=start_token.column,
        )

    # ========================================================
    # Expressions
    # ========================================================

    def parse_expression(self):
        return self.parse_additive()

    def parse_additive(self):
        left = self.parse_multiplicative()

        while self.peek("PLUS") or self.peek("MINUS"):
            operator_token = self.advance()
            operator = operator_token.lexeme
            right = self.parse_multiplicative()

            left = BinaryOp(
                operator,
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_multiplicative(self):
        left = self.parse_unary()

        while self.peek("STAR") or self.peek("SLASH"):
            operator_token = self.advance()
            operator = operator_token.lexeme
            right = self.parse_unary()

            left = BinaryOp(
                operator,
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_unary(self):
        if self.peek("MINUS"):
            operator_token = self.advance()

            return UnaryOp(
                operator_token.lexeme,
                self.parse_unary(),
                line=operator_token.line,
                column=operator_token.column,
            )

        return self.parse_primary()

    def parse_primary(self):
        if self.peek("KW_LENGTH"):
            token = self.advance()
            self.expect("LPAREN")

            arguments = []
            if not self.peek("RPAREN"):
                arguments.append(self.parse_expression())
                while self.peek("COMMA"):
                    self.advance()
                    arguments.append(self.parse_expression())

            self.expect("RPAREN")
            return FunctionCall(
                "length",
                arguments,
                line=token.line,
                column=token.column,
            )

        if self.peek("INT"):
            token = self.advance()

            return IntLiteral(
                int(token.lexeme),
                line=token.line,
                column=token.column,
            )

        if self.peek("IDENT"):
            token = self.advance()
            name = token.lexeme

            if self.peek("LPAREN"):
                return self.parse_function_call_after_name(name, token)

            if self.peek("LBRACKET"):
                self.advance()

                index = self.parse_expression()

                self.expect("RBRACKET")

                return ArrayAccess(
                    name,
                    index,
                    line=token.line,
                    column=token.column,
                )

            return Variable(
                name,
                line=token.line,
                column=token.column,
            )

        if self.peek("LPAREN"):
            self.advance()

            expression = self.parse_expression()

            self.expect("RPAREN")

            return expression

        token = self.current

        if token is None:
            raise ParseError(
                "expected expression, got end of input"
            )

        raise ParseError(
            f"expected expression, got {token.lexeme!r}",
            token,
        )

    # --------------------------------------------------------
    # Function call
    # --------------------------------------------------------

    def parse_function_call(self):
        token = self.expect_identifier()
        return self.parse_function_call_after_name(token.lexeme, token)

    def parse_function_call_after_name(self, name, start_token=None):
        if start_token is None:
            start_token = self.current
        self.expect("LPAREN")

        arguments = []

        if not self.peek("RPAREN"):
            arguments.append(self.parse_expression())

            while self.peek("COMMA"):
                self.advance()
                arguments.append(self.parse_expression())

        self.expect("RPAREN")

        return FunctionCall(
            name=name,
            arguments=arguments,
            line=start_token.line if start_token is not None else None,
            column=start_token.column if start_token is not None else None,
        )

    # ========================================================
    # Conditions
    # ========================================================

    def parse_condition(self):
        return self.parse_implication()

    def parse_implication(self):
        left = self.parse_or()

        if self.peek("ARROW"):
            operator_token = self.advance()

            right = self.parse_implication()

            return PredicateBinary(
                "->",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_or(self):
        left = self.parse_and()

        while self.peek("KW_OR"):
            operator_token = self.advance()

            right = self.parse_and()

            left = PredicateBinary(
                "or",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_and(self):
        left = self.parse_not()

        while self.peek("KW_AND"):
            operator_token = self.advance()

            right = self.parse_not()

            left = PredicateBinary(
                "and",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_not(self):
        if self.peek("KW_NOT"):
            token = self.advance()

            return PredicateNot(
                self.parse_not(),
                line=token.line,
                column=token.column,
            )

        return self.parse_condition_primary()

    def parse_condition_primary(self):
        if self.peek("KW_TRUE"):
            token = self.advance()
            return BoolLiteral(True, line=token.line, column=token.column)

        if self.peek("KW_FALSE"):
            token = self.advance()
            return BoolLiteral(False, line=token.line, column=token.column)

        if self.peek("LPAREN"):
            self.advance()

            result = self.parse_condition()

            self.expect("RPAREN")

            return result

        return self.parse_comparison()

    # --------------------------------------------------------
    # comparison
    # --------------------------------------------------------

    def parse_comparison(self):
        left = self.parse_expression()
        start_line = getattr(left, "line", None)
        start_column = getattr(left, "column", None)

        operators = {
            "EQ",
            "NE",
            "LE",
            "GE",
            "LT",
            "GT",
        }

        if self.current is None:
            raise ParseError(
                "expected comparison operator, "
                "got end of input"
            )

        if self.current.type not in operators:
            raise ParseError(
                f"expected comparison operator, "
                f"got {self.current.lexeme!r}",
                self.current,
            )

        operator = self.advance().lexeme

        right = self.parse_expression()

        return Comparison(
            operator,
            left,
            right,
            line=start_line,
            column=start_column,
        )

    # ========================================================
    # Predicates
    # ========================================================

    def parse_predicate(self):
        return self.parse_predicate_implication()

    def parse_predicate_implication(self):
        left = self.parse_predicate_or()

        if self.peek("ARROW"):
            operator_token = self.advance()

            right = self.parse_predicate_implication()

            return PredicateBinary(
                "->",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_predicate_or(self):
        left = self.parse_predicate_and()

        while self.peek("KW_OR"):
            operator_token = self.advance()

            right = self.parse_predicate_and()

            left = PredicateBinary(
                "or",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_predicate_and(self):
        left = self.parse_predicate_not()

        while self.peek("KW_AND"):
            operator_token = self.advance()

            right = self.parse_predicate_not()

            left = PredicateBinary(
                "and",
                left,
                right,
                line=operator_token.line,
                column=operator_token.column,
            )

        return left

    def parse_predicate_not(self):
        if self.peek("KW_NOT"):
            token = self.advance()

            return PredicateNot(
                self.parse_predicate_not(),
                line=token.line,
                column=token.column,
            )

        return self.parse_predicate_primary()

    def parse_predicate_primary(self):
        if self.peek("KW_FORALL") or self.peek("KW_EXISTS"):
            return self.parse_quantifier()

        if self.peek("KW_TRUE"):
            token = self.advance()
            return BoolLiteral(True, line=token.line, column=token.column)

        if self.peek("KW_FALSE"):
            token = self.advance()
            return BoolLiteral(False, line=token.line, column=token.column)

        if self.peek("LPAREN"):
            self.advance()

            result = self.parse_predicate()

            self.expect("RPAREN")

            return result

        if (
            self.peek("IDENT")
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == "LPAREN"
        ):
            start_token = self.advance()
            name = start_token.lexeme

            self.expect("LPAREN")

            arguments = []

            if not self.peek("RPAREN"):
                arguments.append(self.parse_expression())

                while self.peek("COMMA"):
                    self.advance()
                    arguments.append(self.parse_expression())

            self.expect("RPAREN")

            return FormulaRef(
                name,
                arguments,
                line=start_token.line,
                column=start_token.column,
            )

        return self.parse_comparison()

    # --------------------------------------------------------
    # Quantifier
    # --------------------------------------------------------

    def parse_quantifier(self):
        token = self.advance()

        if token.type == "KW_FORALL":
            kind = "forall"
        else:
            kind = "exists"

        self.expect("LPAREN")

        variable = self.parse_variable_def()

        self.expect("PIPE")

        body = self.parse_predicate()

        self.expect("RPAREN")

        return Quantifier(
            kind,
            variable,
            body,
            line=token.line,
            column=token.column,
        )


# ============================================================
# Public helper
# ============================================================

def parse(tokens) -> Program:
    parser = Parser(tokens)
    program = parser.parse()

    if parser.current is not None:
        token = parser.current

        raise ParseError(
            f"unexpected token {token.lexeme!r}",
            token,
        )

    return program

import json
from dataclasses import asdict, is_dataclass


from funny_lexer import TOKEN_SPECS, build_lexer

# ============================================================
# AST -> JSON
# ============================================================

def ast_to_dict(value):

    if is_dataclass(value):
        result = {
            "node": value.__class__.__name__
        }

        for field in value.__dataclass_fields__:
            result[field] = ast_to_dict(
                getattr(value, field)
            )

        return result

    if isinstance(value, list):
        return [ast_to_dict(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): ast_to_dict(item)
            for key, item in value.items()
        }

    return value


# ============================================================
# AST -> human-readable text
# ============================================================

def ast_to_text(value, indent=0):

    prefix = " " * indent

    if is_dataclass(value):
        lines = [
            f"{prefix}{value.__class__.__name__}"
        ]

        for field in value.__dataclass_fields__:
            field_value = getattr(value, field)

            if field_value is None:
                continue

            lines.append(
                f"{prefix}  {field}:"
            )

            lines.append(
                ast_to_text(
                    field_value,
                    indent + 4
                )
            )

        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"

        return "\n".join(
            ast_to_text(item, indent)
            for item in value
        )

    return f"{prefix}{value!r}"


# ============================================================
# Test helpers
# ============================================================

def parse_source(dfa, source):

    tokens = dfa.tokenize(source)

    parser = Parser(tokens)

    return parser.parse()


def make_positive_test(
    name,
    source,
    dfa,
):

    try:
        ast = parse_source(dfa, source)

        return {
            "name": name,
            "type": "positive",
            "input": source,
            "expected": {
                "result": "AST"
            },
            "actual": {
                "result": "AST",
                "ast": ast_to_dict(ast),
                "ast_text": ast_to_text(ast),
            },
            "status": "PASS",
        }

    except Exception as exc:
        return {
            "name": name,
            "type": "positive",
            "input": source,
            "expected": {
                "result": "AST"
            },
            "actual": {
                "result": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "status": "FAIL",
        }


def make_negative_test(
    name,
    source,
    dfa,
):

    try:
        ast = parse_source(dfa, source)

        return {
            "name": name,
            "type": "negative",
            "input": source,
            "expected": {
                "result": "ERROR"
            },
            "actual": {
                "result": "AST",
                "ast": ast_to_dict(ast),
            },
            "status": "FAIL",
        }

    except Exception as exc:
        return {
            "name": name,
            "type": "negative",
            "input": source,
            "expected": {
                "result": "ERROR"
            },
            "actual": {
                "result": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "status": "PASS",
        }


# ============================================================
# Parser tests
# ============================================================

def run_parser_tests(dfa):
    tests = []

    tests.append(
        make_positive_test(
            "simple_arithmetic",
            """\
main() returns r:int {
    r = 1 + 2 * 3;
}
""",
            dfa,
        )
    )

    tests.append(
        make_positive_test(
            "function_call",
            """\
add(a:int, b:int) returns r:int {
    r = a + b;
}

main() returns r:int {
    r = add(2, 3);
}
""",
            dfa,
        )
    )

    tests.append(
        make_positive_test(
            "multiple_return_values",
            """\
split(x:int) returns a:int, b:int {
    a = x / 2;
    b = x - a;
}

addpair(p:int, q:int) returns s:int {
    s = p + q;
}

main() returns r:int {
    p, q = split(9);
    r = addpair(p, q);
}
""",
            dfa,
        )
    )

    tests.append(
        make_positive_test(
            "control_flow",
            """\
main() returns r:int {
    r = 0;

    if (r == 0) {
        assert r == 0;
    } else {
        assume r > 0;
    }

    while (r < 10)
        invariant r >= 0
    {
        r = r + 1;
    }
}
""",
            dfa,
        )
    )

    tests.append(
        make_positive_test(
            "array_access",
            """\
main(a:int[]) returns r:int {
    a[0] = 10;
    r = a[0] + 2;
}
""",
            dfa,
        )
    )

    tests.append(
        make_negative_test(
            "unmatched_parenthesis",
            """\
main() returns r:int {
    r = (1 + 2;
}
""",
            dfa,
        )
    )

    tests.append(
        make_negative_test(
            "extra_tokens",
            """\
main() returns r:int {
    r = 1;
}

garbage
""",
            dfa,
        )
    )

    tests.append(
        make_negative_test(
            "wrong_operator",
            """\
main() returns r:int {
    r = 1 == 2;
}
""",
            dfa,
        )
    )

    tests.append(
        make_negative_test(
            "empty_input",
            "",
            dfa,
        )
    )

    return tests


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tests",
        default="parser_test_results.json",
        help="JSON file with parser test results",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Build lexer.
    # --------------------------------------------------------

    _, _, dfa = build_lexer(
        list(TOKEN_SPECS)
    )

    # --------------------------------------------------------
    # Run parser tests.
    # --------------------------------------------------------

    report = run_parser_tests(dfa)

    passed = sum(
        item["status"] == "PASS"
        for item in report
    )

    total = len(report)

    # --------------------------------------------------------
    # Save detailed JSON report.
    # --------------------------------------------------------

    output = {
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": total - passed,
        "status": "PASS" if passed == total else "FAIL",
        "tests": report,
    }

    with open(
        args.tests,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output.
    # --------------------------------------------------------

    print(
        f"Parser tests: {passed}/{total} PASS"
    )

    for test in report:
        print(
            f"[{test['status']}] "
            f"{test['name']}"
        )

        if test["status"] == "FAIL":
            print(
                f"    {test['actual']}"
            )

    print(
        f"Test report: {args.tests}"
    )

#     ast = parse_source(dfa, """\
# main() returns r:int {
#     r = 1 + 2;
# }
# """)
#     print(ast_to_text(ast))
if __name__ == "__main__":
    main()