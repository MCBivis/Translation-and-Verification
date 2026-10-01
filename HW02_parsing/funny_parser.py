from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# AST
# ============================================================

@dataclass
class Program:
    functions: list["Function"]


@dataclass
class Function:
    name: str
    parameters: list["VariableDef"]
    returns: list["VariableDef"]
    requires: Optional["Predicate"]
    ensures: Optional["Predicate"]
    uses: list["LocalVarDef"]
    body: "Statement"


@dataclass
class VariableDef:
    name: str
    type: "Type"


@dataclass
class LocalVarDef:
    name: str
    type: Optional["Type"]


@dataclass
class Type:
    name: str
    is_array: bool = False


# ============================================================
# Statements
# ============================================================

class Statement:
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

class Expression:
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

class Predicate:
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
                f"expected {token_type}, got {token.type} "
                f"({token.lexeme!r})",
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
                f"expected identifier, got "
                f"{token.type} ({token.lexeme!r})",
                token,
            )

        self.pos += 1
        return token

    # --------------------------------------------------------
    # Entry point
    # --------------------------------------------------------

    def parse(self) -> Program:
        functions = []

        while self.current is not None:
            functions.append(self.parse_function())

        return Program(functions)

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
        name = self.expect_identifier().lexeme

        self.expect("COLON")

        type_ = self.parse_type()

        return VariableDef(name, type_)

    def parse_local_var_def(self):
        name = self.expect_identifier().lexeme

        type_ = None

        if self.peek("COLON"):
            self.advance()
            type_ = self.parse_type()

        return LocalVarDef(name, type_)

    # --------------------------------------------------------
    # type
    # --------------------------------------------------------

    def parse_type(self):
        self.expect("KW_INT")

        if self.peek("LBRACKET"):
            self.advance()
            self.expect("RBRACKET")

            return Type("int", is_array=True)

        return Type("int")

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
            f"expected statement, got "
            f"{token.type} ({token.lexeme!r})",
            token,
        )

    # --------------------------------------------------------
    # block
    # --------------------------------------------------------

    def parse_block(self):
        self.expect("LBRACE")

        statements = []

        while self.current is not None and not self.peek("RBRACE"):
            statements.append(self.parse_statement())

        self.expect("RBRACE")

        return Block(statements)

    # --------------------------------------------------------
    # if
    # --------------------------------------------------------

    def parse_if(self):
        self.expect("KW_IF")
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
        )

    # --------------------------------------------------------
    # while
    # --------------------------------------------------------

    def parse_while(self):
        self.expect("KW_WHILE")
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
        )

    # --------------------------------------------------------
    # assert
    # --------------------------------------------------------

    def parse_assert(self):
        self.expect("KW_ASSERT")

        predicate = self.parse_predicate()

        self.expect("SEMICOLON")

        return AssertStatement(predicate)

    # --------------------------------------------------------
    # assume
    # --------------------------------------------------------

    def parse_assume(self):
        self.expect("KW_ASSUME")

        predicate = self.parse_predicate()

        self.expect("SEMICOLON")

        return AssumeStatement(predicate)

    # --------------------------------------------------------
    # assignment
    # --------------------------------------------------------

    def parse_assignment(self):
        first = self.expect_identifier().lexeme

        # ----------------------------------------------------
        # Tuple assignment:
        #
        # p, q = split(9);
        # ----------------------------------------------------

        if self.peek("COMMA"):
            targets = [Variable(first)]

            while self.peek("COMMA"):
                self.advance()

                name = self.expect_identifier().lexeme
                targets.append(Variable(name))

            self.expect("ASSIGN")

            value = self.parse_function_call()

            self.expect("SEMICOLON")

            return Assignment(targets, value)

        # ----------------------------------------------------
        # Single assignment.
        # ----------------------------------------------------

        target = Variable(first)

        if self.peek("LBRACKET"):
            self.advance()

            index = self.parse_expression()

            self.expect("RBRACKET")

            target = ArrayAccess(first, index)

        self.expect("ASSIGN")

        value = self.parse_expression()

        self.expect("SEMICOLON")

        return Assignment([target], value)

    # ========================================================
    # Expressions
    # ========================================================

    def parse_expression(self):
        return self.parse_additive()

    def parse_additive(self):
        left = self.parse_multiplicative()

        while self.peek("PLUS") or self.peek("MINUS"):
            operator = self.advance().lexeme
            right = self.parse_multiplicative()

            left = BinaryOp(
                operator,
                left,
                right,
            )

        return left

    def parse_multiplicative(self):
        left = self.parse_unary()

        while self.peek("STAR") or self.peek("SLASH"):
            operator = self.advance().lexeme
            right = self.parse_unary()

            left = BinaryOp(
                operator,
                left,
                right,
            )

        return left

    def parse_unary(self):
        if self.peek("MINUS"):
            operator = self.advance().lexeme

            return UnaryOp(
                operator,
                self.parse_unary(),
            )

        return self.parse_primary()

    def parse_primary(self):
        if self.peek("INT"):
            token = self.advance()

            return IntLiteral(int(token.lexeme))

        if self.peek("IDENT"):
            name = self.advance().lexeme

            if self.peek("LPAREN"):
                return self.parse_function_call_after_name(name)

            if self.peek("LBRACKET"):
                self.advance()

                index = self.parse_expression()

                self.expect("RBRACKET")

                return ArrayAccess(name, index)

            return Variable(name)

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
            f"expected expression, got "
            f"{token.type} ({token.lexeme!r})",
            token,
        )

    # --------------------------------------------------------
    # Function call
    # --------------------------------------------------------

    def parse_function_call(self):
        name = self.expect_identifier().lexeme

        return self.parse_function_call_after_name(name)

    def parse_function_call_after_name(self, name):
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
        )

    # ========================================================
    # Conditions
    # ========================================================

    def parse_condition(self):
        return self.parse_implication()

    def parse_implication(self):
        left = self.parse_or()

        if self.peek("ARROW"):
            self.advance()

            right = self.parse_implication()

            return PredicateBinary(
                "->",
                left,
                right,
            )

        return left

    def parse_or(self):
        left = self.parse_and()

        while self.peek("KW_OR"):
            self.advance()

            right = self.parse_and()

            left = PredicateBinary(
                "or",
                left,
                right,
            )

        return left

    def parse_and(self):
        left = self.parse_not()

        while self.peek("KW_AND"):
            self.advance()

            right = self.parse_not()

            left = PredicateBinary(
                "and",
                left,
                right,
            )

        return left

    def parse_not(self):
        if self.peek("KW_NOT"):
            self.advance()

            return PredicateNot(
                self.parse_not()
            )

        return self.parse_condition_primary()

    def parse_condition_primary(self):
        if self.peek("KW_TRUE"):
            self.advance()
            return BoolLiteral(True)

        if self.peek("KW_FALSE"):
            self.advance()
            return BoolLiteral(False)

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
                f"got {self.current.type} "
                f"({self.current.lexeme!r})",
                self.current,
            )

        operator = self.advance().lexeme

        right = self.parse_expression()

        return Comparison(
            operator,
            left,
            right,
        )

    # ========================================================
    # Predicates
    # ========================================================

    def parse_predicate(self):
        return self.parse_predicate_implication()

    def parse_predicate_implication(self):
        left = self.parse_predicate_or()

        if self.peek("ARROW"):
            self.advance()

            right = self.parse_predicate_implication()

            return PredicateBinary(
                "->",
                left,
                right,
            )

        return left

    def parse_predicate_or(self):
        left = self.parse_predicate_and()

        while self.peek("KW_OR"):
            self.advance()

            right = self.parse_predicate_and()

            left = PredicateBinary(
                "or",
                left,
                right,
            )

        return left

    def parse_predicate_and(self):
        left = self.parse_predicate_not()

        while self.peek("KW_AND"):
            self.advance()

            right = self.parse_predicate_not()

            left = PredicateBinary(
                "and",
                left,
                right,
            )

        return left

    def parse_predicate_not(self):
        if self.peek("KW_NOT"):
            self.advance()

            return PredicateNot(
                self.parse_predicate_not()
            )

        return self.parse_predicate_primary()

    def parse_predicate_primary(self):
        if self.peek("KW_FORALL") or self.peek("KW_EXISTS"):
            return self.parse_quantifier()

        if self.peek("KW_TRUE"):
            self.advance()
            return BoolLiteral(True)

        if self.peek("KW_FALSE"):
            self.advance()
            return BoolLiteral(False)

        if self.peek("LPAREN"):
            self.advance()

            result = self.parse_predicate()

            self.expect("RPAREN")

            return result

        # identifier(...) в предикате трактуем
        # как ссылку на формулу.
        if (
            self.peek("IDENT")
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == "LPAREN"
        ):
            name = self.advance().lexeme

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
            f"unexpected token "
            f"{token.type} ({token.lexeme!r})",
            token,
        )

    return program