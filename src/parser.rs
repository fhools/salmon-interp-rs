use crate::lex::{Token, TokenType};
use crate::LoxToken;
use super::expr::*;
use crate::error::ParseError;
use super::salmon_error;
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

    fn consume(&mut self, tok_type: &TokenType, expected_str: &'static str) -> Result<(), ParseError> {
        if self.check(tok_type) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::General(self.peek().clone(), expected_str.to_string()))
        }
    }

    fn parse(&mut self) -> Result<Box<Expr>, ParseError> {
       let expr = self.expression();
       let mut errorVisitor = ErrorVisitor::new();
       errorVisitor.visit_expr(&expr);
       if errorVisitor.has_error_node() {
           let mut print_visitor = PrintVisitor{};
           let err_print = print_visitor.visit_expr(&expr);
           println!("err print: {}", err_print);
           Err(ParseError::General(self.previous().clone(), format!("err ast: {}", err_print)))
       } else {
           Ok(expr)
       }
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
                    _ => { Expr::ParseError }
            };
            Box::new(expr)
        } else if self.match_any_of(&[LoxToken![LeftParen]]) {
            let expr = self.expression();
            let result = self.consume(&LoxToken![RightParen], "expected ')' after expression");
            match result {
                Ok(_) => {
                    Box::new(Expr::Grouping(GroupingExpr{ group: Box::new(*expr) }))
                },
                Err(_) => {
                    Box::new(Expr::ParseError)
                }
            }
        } else {
            let tok = self.peek().to_owned();
            self.error(&tok, "parse primary fail");
            Box::new(Expr::ParseError)
        }
    }

    fn error(&mut self, tok: &Token, msg: impl Into<String>) {
        let m = format!("parse error on token type: {:?} msg: {}", tok.token_type, msg.into());
        salmon_error(tok.line, m.as_str());
    }

    fn synchronize(&mut self) {
        // skip over bad token
        self.advance();
        while !self.is_at_end() {
            if self.previous().token_type == TokenType::Semicolon {
                return;
            }
            // sychronization points
            match self.peek().token_type {
                TokenType::Class | TokenType::For | TokenType::Fun |
                TokenType::If | TokenType::Print | TokenType::Return |
                TokenType::Var | TokenType::While => 
                { return; },
                _ => {}
            }
            // keep consuming tokens 
            self.advance();
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

    fn do_expr(source: impl Into<String>) -> Result<Box<Expr>, ParseError> {
        let mut parser = Parser::new(&gen_tokens(source.into().as_str()));
        parser.parse()
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

    #[test]
    fn test_parser_parse() {
        let mut parser = Parser::new(&gen_tokens("1+2"));
        let expr = parser.parse();
        if expr.is_ok() {
            let print_output  = expr.as_ref().map(|e| {
                let mut print_visitor = PrintVisitor{};
                print_visitor.visit_expr(&e)
            }).unwrap_or("issue".to_string());
            eprintln!("parse() returned: {}", print_output);
        }
        assert!(expr.is_ok());
    }

    #[test]
    fn test_parser_parse_bad() {
        let mut parser = Parser::new(&gen_tokens("1+"));
        let expr = parser.parse();
        if expr.is_ok() {
            let print_output  = expr.as_ref().map(|e| {
                let mut print_visitor = PrintVisitor{};
                print_visitor.visit_expr(&e)
            }).unwrap_or("issue".to_string());
            eprintln!("parse_bad test - parse() returned: {}", print_output);
        } else {
            eprintln!("parse failed as expected");
        }
    }
    #[test]
    fn test_parse_binary_expr2() {
        let expr = do_expr("1+2+3")
            .map(|expr| expr.to_string())
            .unwrap_or_else(|_| {
                "error".to_string()
            });
        assert_eq!(expr, "(+ (+ 1 2) 3)");
    }
}

