use crate::lex::{Token, TokenType, gen_tokens};
use crate::LoxToken;
use super::expr::*;
use crate::error::ParseError;
use super::salmon_error;
pub(crate) struct Parser {
    tokens: Vec<Token>,

    // index into tokens vector
    current: usize,

    // error synchronization flag
    need_sync: bool,
}

impl Parser {

    pub fn new(tokens: &Vec<Token>) -> Self {
        Parser {
            tokens: tokens.clone(),
            current: 0,
            need_sync: false
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

    // matches any of the provided tokens, and advances the cursor
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

    fn consume(&mut self, tok_type: &TokenType, expected_str: &'static str) -> Result<&Token, ParseError> {
        if self.check(tok_type) {
            Ok(self.advance())
        } else {
            eprintln!("consume err: {}", expected_str);
            Err(ParseError::General(self.peek().clone(), expected_str.to_string()))
        }
    }


    // converts an Expr into a Result<Box<Expr>, ParseError> by 
    // checking the expr for any ParseError nodes
    fn check_parse_error(&self, expr: Expr) -> Result<Box<Expr>, ParseError> {
       let mut errorVisitor = ErrorVisitor::new();
       errorVisitor.visit_expr(&expr);
       if errorVisitor.has_error_node() {
           let mut print_visitor = PrintVisitor{};
           let err_print = print_visitor.visit_expr(&expr);
           println!("err print: {}", err_print);
           Err(ParseError::General(self.previous().clone(), format!("err ast: {}", err_print)))
       } else {
           Ok(Box::new(expr))
       }
    }


    pub fn parse(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let stmts = self.parse_program();
        Ok(stmts)
    }

    // program := declaration* EOF
    fn parse_program(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while !self.is_at_end() {
            stmts.push(self.declaration());
        }
        stmts
    }


    // delaration := vardecl 
    //              | statement 
    fn declaration(&mut self) -> Stmt {
        let stmt;
        if self.match_any_of(&[LoxToken![Var]]) {
            stmt = self.var_declaration();
            if self.need_sync {
                self.synchronize();
            }
        } else {
            stmt = self.statement();
            if self.need_sync {
                self.synchronize();
            }
        }
        stmt
    }
   
    // var_declaration := VAR "=" expression
    
    fn var_declaration(&mut self) -> Stmt {
        // TODO: Note this is a kludge, consume's check of Identifier does not check the value of identifier so it
        // works
        let name = match self.consume(&LoxToken![Identifier("".to_string())], "expected identifier after var") {
            Ok(token) => {
                token.to_owned()
            },
            Err(_) => {
                Token::new_dummy_identifier(self.previous().line)
            }
        };
        let var_decl_stmt = if self.match_any_of(&[LoxToken![Equal]])  {
            Stmt::VarDecl(VarDecl{name, initializer: Some(self.expression())})
        } else {
            Stmt::VarDecl(VarDecl{name, initializer: None})
        };
        if self.consume(&LoxToken![Semicolon], "expected ';' at the end of var decl").is_err() {
            self.set_need_sync();
        };
        var_decl_stmt
    }

    // statement := "print" expression_statement
    //             | expression_statement
    fn statement(&mut self) -> Stmt {
        if self.match_any_of(&[LoxToken![Print]]) {
            self.print_statement()
        } else {
            self.expression_statement()
        }
    }

    // print_statement := expression ';'
    fn print_statement(&mut self) -> Stmt {
        let expr = self.expression(); 
        if self.consume(&TokenType::Semicolon, "expect semicolon after print statement").is_err() {
            self.set_need_sync();
        }
        Stmt::Print(expr)
    }
    
    // expression_statement := expression ";"
    fn expression_statement(&mut self) -> Stmt {
        let expr = self.expression();
        if self.consume(&TokenType::Semicolon, "expect semicolon after expression statement").is_err() {
            self.set_need_sync();
        } 
        Stmt::Expression(expr)
    }

    // expression := assignment ";"
    fn expression(&mut self) -> Box<Expr> {
        self.assignment()
    }

    // IDENTIFIER but really a VariableExpr
    // assignment := IDENTIFIER = assignment
    //              | equality
    fn assignment(&mut self) -> Box<Expr> {
        // This is an interesting algorith in the book.
        // We try to parse an expression
        // We then check to see if the next token is = , and if it is then we
        // verify whether the expression we parsed earlier is a VariableExpr
        // if it is then we recursivel parse assignment again
        // if it is not then we have an error we can't assign to non variable expression

        // this could be a var 
        let left_side_expr = self.equality();

        if self.match_any_of(&[LoxToken![Equal]]) {
            let equals_tok = self.previous();
            let value = self.assignment();
            match *left_side_expr {
                Expr::Variable(var) => {
                    Box::new(Expr::Assign(AssignExpr{name: var.name, value}))
                },
                _ => {
                    eprintln!("invalid lvalue for var assignment");
                    Box::new(Expr::ParseError)
                }
            }
        } else {
            left_side_expr
        }
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
            let op = self.previous().clone();
            let expr = self.unary();
            Box::new(Expr::Unary(UnaryExpr{ op, expr }))
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
        } else if self.match_any_of(&[LoxToken![Identifier("".to_string())]]) {
            Box::new(Expr::Variable(VariableExpr{name:self.previous().clone()}))
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
                self.need_sync = false;
                return;
            }
            // sychronization points
            match self.peek().token_type {
                TokenType::Class | TokenType::For | TokenType::Fun |
                TokenType::If | TokenType::Print | TokenType::Return |
                TokenType::Var | TokenType::While => 
                { 
                    self.need_sync = false;
                    return;
                },
                _ => {}
            }
            // keep consuming tokens 
            self.advance();
        }
    }

    fn set_need_sync(&mut self) {
        self.need_sync = true;
    }

    fn clear_need_sync(&mut self) {
        self.need_sync = false;
    }

}

pub fn do_expr(source: impl Into<String>) -> Result<Box<Expr>, ParseError> {
    let mut parser = Parser::new(&gen_tokens(source.into().as_str()));
    let expr = parser.expression();
    parser.check_parse_error(*expr)
}

mod test {

    use super::super::lex::{Scanner, TokenType, gen_tokens};
    use super::*;

    fn create_tokens() -> Vec<Token> {
        let mut scanner = Scanner::new("this is a test");
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

    #[test]
    fn test_parser_parse() {
        let expr = do_expr("1+2"); 
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
        let expr = do_expr("1+");
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
    #[test]
    fn test_parse_unary_expr() {
        let expr = do_expr("1 + 2 + 3 + -4")
            .map(|expr| expr.to_string())
            .unwrap_or_else(|_| {
                "error".to_string()
            });
        assert_eq!(expr, "(+ (+ (+ 1 2) 3) (- 4))");
    }
}

