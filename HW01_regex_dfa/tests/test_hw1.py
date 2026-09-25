import unittest

from funny_lexer import TOKEN_SPECS, build_lexer


class FunnyLexerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.dfa = build_lexer(list(TOKEN_SPECS))

    def test_empty(self):
        self.assertEqual(self.dfa.tokenize(""), [])

    def test_whitespace(self):
        self.assertEqual(self.dfa.tokenize(" \t\r\n"), [])

    def test_keywords(self):
        result = self.dfa.tokenize("function returns while if else assert assume invariant length")
        self.assertEqual([x[0] for x in result],
                         ["KW_FUNCTION", "KW_RETURNS", "KW_WHILE", "KW_IF",
                          "KW_ELSE", "KW_ASSERT", "KW_ASSUME",
                          "KW_INVARIANT", "KW_LENGTH"])

    def test_identifiers(self):
        self.assertEqual(
            self.dfa.tokenize("_x A_1 abc123"),
            [("IDENT", "_x"), ("IDENT", "A_1"), ("IDENT", "abc123")]
        )

    def test_integer(self):
        self.assertEqual(self.dfa.tokenize("0 10 123"), [
            ("INT", "0"), ("INT", "10"), ("INT", "123")
        ])

    def test_delimiters_and_operators(self):
        self.assertEqual(
            self.dfa.tokenize("()[]{},; + - * / == != <= >= < > ="),
            [("LPAREN","("),("RPAREN",")"),("LBRACKET","["),
             ("RBRACKET","]"),("LBRACE","{"),("RBRACE","}"),
             ("COMMA",","),("SEMICOLON",";"),("PLUS","+"),
             ("MINUS","-"),("STAR","*"),("SLASH","/"),("EQ","=="),
             ("NE","!="),("LE","<="),("GE",">="),("LT","<"),
             ("GT",">"),("ASSIGN","=")]
        )

    def test_comment(self):
        self.assertEqual(self.dfa.tokenize("// comment\r\nx"),
                         [("IDENT", "x")])

    def test_non_ascii_is_error(self):
        with self.assertRaises(ValueError):
            self.dfa.tokenize("привет")

    def test_unknown_ascii_is_error(self):
        with self.assertRaises(ValueError):
            self.dfa.tokenize("@")

    def test_longest_match(self):
        self.assertEqual(self.dfa.tokenize("function1"), [("IDENT", "function1")])

    def test_leading_zero_is_not_one_integer_token(self):
        result = self.dfa.tokenize("01")
        self.assertNotEqual(result, [("INT", "01")])


if __name__ == "__main__":
    unittest.main()
