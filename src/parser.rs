use crate::lex::{Token, TokenType};
use crate::{LoxToken};
use super::expr::*;
pub(crate) struct Parser {
    tokens: Vec<Token>,

    // index into tokens vector
    current: usize
}

impl Parser {

    fn new(tokens: &Vec<Token>) -> Self {
        Parser {
            tokens: tokens.clone(),
            current: 0
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn previous(&self) ->  &Token {
        &self.tokens[self.current-1]
    }

    fn match_any_of(&mut self, candidate_toks: &[TokenType]) -> bool {
        for t in candidate_toks {
            if self.check(t) {
                self.advance();
                return true
            }
        }
        false
    }

    fn check(&mut self, tok_type: &TokenType) -> bool {
        if self.is_at_end() {
            return false;
        }
        //self.peek().token_type == *tok_type
        // we're doing a less strict check here, as long as they as its any string
        // or any number or any identifier its ok, the cotents don't need to be equal
        // that would be a later part of the parser
        match (&self.peek().token_type, tok_type) {
            (TokenType::String(_), TokenType::String(_)) => true,
            (TokenType::Number(_), TokenType::Number(_)) => true,
            (TokenType::Identifier(_), TokenType::Identifier(_)) => true,
            (a, b) => *a == *b
        }
    }

    fn is_at_end(&mut self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    fn peek(&mut self) -> &Token {
        &self.tokens[self.current]
    }


    // expression := equality
    fn expression(&mut self) -> Box<Expr> {
        self.equality()

    }

    // equality := comparison ( ( "!=" | "==" ) comparison )*
    fn equality(&mut self) -> Box<Expr> {
        let mut expr = self.comparison();
        while self.match_any_of(&[LoxToken![BangEqual], LoxToken![EqualEqual]]) {
            let op = self.previous().clone();
            let right_expr = self.comparison();
            expr = Box::new(Expr::Binary(BinaryExpr{ left: expr, op, right: right_expr}));
        }
        expr
    }

    // comparison := term ( (">" | ">=" | "<" | "<=") term)*
    fn comparison(&mut self) -> Box<Expr> {
        let mut expr = self.term();
        while self.match_any_of(&[LoxToken![Greater], 
                                  LoxToken![GreaterEqual], 
                                  LoxToken![Less], LoxToken![LessEqual]]) {
            let op = self.previous().clone();
            let right_expr = self.term();
            expr = Box::new(Expr::Binary(BinaryExpr{ left: expr, op, right: right_expr}));
        }
        expr
    }

    // term := factor ( ("-" | "+" ) factor)*
    fn term(&mut self) -> Box<Expr> {
        let mut expr = self.factor();
        while self.match_any_of(&[LoxToken![Plus], 
                                  LoxToken![Minus]]) {
            let op = self.previous().clone();
            let right_expr = self.factor();
            expr = Box::new(Expr::Binary(BinaryExpr{ left: expr, op, right: right_expr}));
        }
        expr
    }

    // factor := unary ( ("/" | "*" ) unary)*
    fn factor(&mut self) -> Box<Expr> {
        let mut expr = self.unary();
        while self.match_any_of(&[LoxToken![Slash], 
                                  LoxToken![Star]]) {
            let op = self.previous().clone();
            let right_expr = self.unary();
            expr = Box::new(Expr::Binary(BinaryExpr{ left: expr, op, right: right_expr}));
        }
        expr
    }

    // unary := ("!" | "-") unary 
    //        |  primary
    fn unary(&mut self) -> Box<Expr> {
        if self.match_any_of(&[LoxToken![Bang], LoxToken![Minus]]) {
            return self.unary();
        } else {
            return self.primary();
        }
    }

    // primary := NUMBER | STRING | "true" | "false" | "nil"
    fn primary(&mut self) -> Box<Expr> {
        let num_tok = TokenType::Number(0.0);
        let str_tok = TokenType::String(String::new());
        if self.match_any_of(&[num_tok, str_tok, LoxToken![True], LoxToken![False], LoxToken![Nil]]) {
            let expr = match self.previous().token_type {
                TokenType::Number(n) => 
                    Expr::Literal(LiteralExpr{ val: LoxValue::Number(n) }),
                TokenType::String(ref s) =>
                    Expr::Literal(LiteralExpr{ val: LoxValue::String(s.clone()) }),
                TokenType::True => 
                    Expr::Literal(LiteralExpr{ val: LoxValue::Bool(true) }),
                TokenType::False => 
                    Expr::Literal(LiteralExpr{ val: LoxValue::Bool(false) }),
                TokenType::Nil => 
                    Expr::Literal(LiteralExpr{ val: LoxValue::Nil}),
                    _ => { panic!("expected literal");} 
            };
            Box::new(expr)
        } else if self.match_any_of(&[LoxToken![LeftParen]]) {
            let expr = self.expression();
            self.match_any_of(&[LoxToken![RightParen]]);
            expr
        } else {
            panic!("could not match primary");
        }
    }
}

mod test {

    use super::super::lex::{Scanner, TokenType};
    use super::*;

    fn create_tokens() -> Vec<Token> {
        let mut scanner = Scanner::new("this is a test");
        scanner.scan_tokens()
    }

    fn gen_tokens<S: Into<String>>(source: S) -> Vec<Token> {
        let mut scanner = Scanner::new(source.into());
        scanner.scan_tokens()
    }

    fn num_tok(n: f32) -> Token {
        Token::new(TokenType::Number(n), n.to_string(), 1)
    }

    #[test]
    fn test_parse_cursor() {
        let mut parser = Parser::new(&create_tokens());
        let t = parser.peek();
        assert_eq!(t.token_type, TokenType::This);
        parser.advance();
        let t = parser.peek();
        eprintln!("token peeked: {:?}", t);
        let ident_matched = parser.match_any_of(&[LoxToken![Identifier("".to_string())]]); 
        assert!(ident_matched)
    }

    #[test]
    fn test_parser_match() {
        let mut parser = Parser::new(&gen_tokens("1 2 3 4"));
        let matched = parser.match_any_of(&[num_tok(1.0).token_type]);
        assert!(matched);
    }
}

