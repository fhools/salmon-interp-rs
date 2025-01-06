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

    pub fn new(tokens: &[Token]) -> Self {
        Parser {
            tokens: tokens.to_vec(),
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

    // peek check to see if next token is tok_type
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

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn consume(&mut self, tok_type: &TokenType, expected_str: &str) -> Result<&Token, ParseError> {
        if self.check(tok_type) {
            Ok(self.advance())
        } else {
            //eprintln!("consume err: {}", expected_str);
            Err(ParseError::General(self.peek().clone(), expected_str.to_string()))
        }
    }


    // converts an Expr into a Result<Box<Expr>, ParseError> by 
    // checking the expr for any ParseError nodes
    fn check_parse_error(&self, expr: Expr) -> Result<Box<Expr>, ParseError> {
       let mut error_visitor = ErrorVisitor::new();
       error_visitor.visit_expr(&expr);
       if error_visitor.has_error_node() {
           let mut print_visitor = PrintVisitor{};
           let err_print = print_visitor.visit_expr(&expr);
           println!("err print: {}", err_print);
           Err(ParseError::General(self.peek().clone(), format!("err ast: {}", err_print)))
       } else {
           Ok(Box::new(expr))
       }
    }

    fn check_parse_error_stmt(&self, stmt: Stmt) -> Result<Box<Stmt>, ParseError> {
       let mut error_visitor = ErrorVisitor::new();
       error_visitor.visit_stmt(&stmt);
       let mut print_visitor = PrintVisitor{};
       let out = print_visitor.visit_stmt(&stmt);
        eprintln!("stmt ast: {}", out);
       if error_visitor.has_error_node() {
           let mut print_visitor = PrintVisitor{};
           let err_print = print_visitor.visit_stmt(&stmt);
           println!("err print: {}", err_print);
           Err(ParseError::General(self.peek().clone(), format!("err ast: {}", err_print)))
       } else {
           Ok(Box::new(stmt))
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


    // delaration := class_decl 
    //              | vardecl 
    //              | function_decl
    //              | statement 
    fn declaration(&mut self) -> Stmt {
        let stmt;
        if self.match_any_of(&[LoxToken![Class]]) {
            stmt = self.class_decl();
        } else if self.match_any_of(&[LoxToken![Fun]]) {
            stmt = self.function("function");
        } else if self.match_any_of(&[LoxToken![Var]]) {
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
  
    // class_decl := "class" IDENTIFIER "{" function* "}"
    fn class_decl(&mut self) -> Stmt {

        // need to unwrap and clone the token, for the borrow checker
        let class_name =  match self.consume(&LoxToken![Identifier("".to_string())], "expected class name") {
            Ok(class_name) => class_name.clone(),
            Err(_) => return Stmt::ParseError
        };
        self.consume(&LoxToken![LeftBrace], "expecting '{{' before class body");

        let mut methods = Vec::new();

        // NOTE: need to assign results of check into a temp block to satisfy borrow checker, so
        // that the borrow by check() does not last the whole body
        while { let not_right_paren = !self.check(&LoxToken![RightBrace]); 
            not_right_paren }  {
                if let Stmt::Function(ref function_stmt) = self.function("method") {
                methods.push(function_stmt.clone());
            } else {
                let tok_line = self.previous().clone();
                salmon_error(tok_line.line, "unexpected statements in class body, expected methods");
            }
        }
        self.consume(&LoxToken![RightBrace], "expected '}' after class body ");
        Stmt::Class(ClassStmt{ name: class_name, methods})
    }
    
    // function_decl := 'fun' function 
    fn function(&mut self, kind: &str) -> Stmt {
        // TODO: we are using unwrap() here on the self.consume() calls to get the Token. 
        //  this points to a design issue where none of our parsing functions are failable
        //  so we can't use consume()? style
        let mut params = Vec::new();
        let name = self.consume(&LoxToken![Identifier("".to_string())], &format!("expecting {} name", kind)).unwrap().clone();
        self.consume(&LoxToken![LeftParen], &format!("expecting '(' after {} name", kind)); 
        if !self.check(&LoxToken![RightParen]) {
            loop {
                if params.len() >= 255 {
                    let tok = self.previous().clone();
                    self.error(&tok, "can't have more than 255 parameters");
                }
                let param = self.consume(&LoxToken![Identifier("".to_string())], "expect parameter name");
                params.push(param.unwrap().clone());
                if !self.match_any_of(&[LoxToken![Comma]]) {
                    break;
                }
            }
        }
        self.consume(&LoxToken![RightParen], "expecting ')' after parameters");
        self.consume(&LoxToken![LeftBrace], &format!("expecting '{{' before {} body", kind));
        let body = self.block();
        Stmt::Function(FunctionStmt{name, params, body})
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

    // statement := if_statement 
    //             | while_statement
    //             | for_statement
    //             | "print" expression_statement
    //             | expression_statement
    //             | block 
    //             | return_statement
    fn statement(&mut self) -> Stmt {
        if self.match_any_of(&[LoxToken![If]]) {
            self.if_statement()
        } else if self.match_any_of(&[LoxToken![While]]) {
            self.while_statement()
        } else if self.match_any_of(&[LoxToken![For]]) {
            self.for_statement()
        } else if self.match_any_of(&[LoxToken![Print]]) {
            self.print_statement()
        } else if self.match_any_of(&[LoxToken![LeftBrace]]) {
            Stmt::Block(Block{statements: self.block()})
        } else if self.match_any_of(&[LoxToken![Return]]) {
            self.return_statement()
        }else {
            self.expression_statement()
        }
    }
    // ifStmt := if "(" expression ")" statement ("else" statement )?
    fn if_statement(&mut self) -> Stmt {
        self.consume(&LoxToken![LeftParen], "expected '(' following 'if'");
        let conditional = self.expression();
        self.consume(&LoxToken![RightParen], "expected ')' following if condtional");
        let then_branch = self.statement();
        let mut else_branch: Option<Box<Stmt>> = None;
        if self.match_any_of(&[LoxToken![Else]]) {
            else_branch = Some(Box::new(self.statement()));
        }
        Stmt::If(IfStmt { conditional, then_branch: Box::new(then_branch) , else_branch })
    }

    // while_statement := while "(" expression ")" statement 
    fn while_statement(&mut self) -> Stmt {
        self.consume(&LoxToken![LeftParen], "expected '(' following 'while'");
        let condition = self.expression();
        self.consume(&LoxToken![RightParen], "expected ')' following 'while'");
        let body = self.statement();
        let while_stmt = Stmt::While(WhileStmt{ condition, body: Box::new(body) });
        while_stmt
    }

    // for_statement := for "(" ( var_decl | expr_stmt | ";" expression? ";" expression? ")"
    //                  statement
    fn for_statement(&mut self) -> Stmt {
        // Some interest stuff about this, var_decl eand expr_stmt both are optional and parsing
        // those will munch the semicolon

        self.consume(&LoxToken![LeftParen], "expected '(' following 'for'");

        let initializer;
        // if the next token is a semicolon that means there is no var/expression initializer
        if self.match_any_of(&[LoxToken![Semicolon]]) {
            initializer =  None
        // otherwise if its a var 
        } else if self.match_any_of(&[LoxToken![Var]]) {
            initializer = Some(self.var_declaration());
        // or its an expression, notice we use the expression_statement which will munch the
        // semilcolon
        } else  {
            initializer = Some(self.expression_statement());
        }
      
        // there is an optional condition, if not present its a while(true) equivalent
        let condition = if !self.check(&LoxToken![Semicolon]) {
            Some(self.expression())
        } else {
            Some(Box::new(new_expr(ExprKind::Literal(LiteralExpr{val: LoxValue::Bool(true)}))))
        };

        self.consume(&LoxToken![Semicolon], "expected ';' after condition for 'for'");

        let mut increment = None;
        if !self.check(&LoxToken![RightParen]) {
            increment = Some(self.expression());
        }
        // finally munch the closing paren
        self.consume(&LoxToken![RightParen], "expected ')' following for clause");

        let mut body = self.statement();

        if let Some(increment) = increment {
            let body_with_incr = vec![body, Stmt::Expression(increment)];
            body = Stmt::Block(Block{ statements: body_with_incr }); 
        }


        body = Stmt::While(WhileStmt { condition: condition.unwrap(), body: Box::new(body)});
    
        if let Some(initializer) = initializer {
            // NOTE: Kind of interesting here the initializer is in it's own block 
            // and then the body of the for loop is another nested block inside that
            // is that an ok design?
            body = Stmt::Block(Block{statements: vec![initializer, body]});
        }
        body
    }

    // block := "{" declaration * "}"
    fn block(&mut self) -> Vec<Stmt> {
        let mut statements = Vec::new();
        while !self.check(&LoxToken![RightBrace]) && !self.is_at_end() {
            statements.push(self.declaration())
        }
        self.consume(&LoxToken![RightBrace], "expected right brace closing block");
        statements 
    }

    // print_statement := expression ';'
    fn print_statement(&mut self) -> Stmt {
        let expr = self.expression(); 
        if self.consume(&TokenType::Semicolon, "expect semicolon after print statement").is_err() {
            self.set_need_sync();
        }
        Stmt::Print(expr)
    }

    // return_statement := "return" expression ";"
    fn return_statement(&mut self) -> Stmt {
        let return_tok = self.previous().clone();
        let mut expr = None;
        if !self.check(&LoxToken![Semicolon]) {
            expr = Some(self.expression());
        }
        self.consume(&LoxToken![Semicolon], "expect ';' after return expression");
        Stmt::Return(ReturnStmt { return_tok, value: expr})
    }

    
    // expression_statement := expression ";"
    fn expression_statement(&mut self) -> Stmt {
        let expr = self.expression();
        if self.consume(&TokenType::Semicolon, "expect semicolon after expression statement").is_err() {
            self.set_need_sync();
        } 
        Stmt::Expression(expr)
    }

    // expression := assignment 
    fn expression(&mut self) -> Box<Expr> {
        self.assignment()
    }

    // IDENTIFIER but really goes parsed as only VariableExpr i.e. lvalue
    // assignment := (call ".")? IDENTIFIER = assignment
    //              | logical_or 
    fn assignment(&mut self) -> Box<Expr> {
        // The implementation of this from the book is notable for the following reasons: 
        // We try to parse an expression
        // We then check to see if the next token is = , and if it is then we
        // verify whether the expression we parsed earlier is a VariableExpr
        // if it is then we recursivel parse assignment again
        // if it is not then we have an error we can't assign to non variable expression
        // if the VariableExpr was not followed by a equal sign then it must be just 
        // expression by itself

        // this could be a var lvalue not just a logical
        let expr = self.logical_or();

        // optionally an = for assignment
        if self.match_any_of(&[LoxToken![Equal]]) {
            let equals_tok = self.previous().clone();
            let value = self.assignment();

            // verify that the expr prior to = is an lvalue
            match &expr.kind {
                ExprKind::Variable(var) => {
                    Box::new(new_expr(ExprKind::Assign(AssignExpr{name: var.name.clone(), value})))
                },

                // if we parsed a Get then transform it into a Set since we saw an equal
                ExprKind::Get(get_expr) => {
                    Box::new(new_expr(ExprKind::Set(SetExpr{
                        object: get_expr.object.clone(), name: get_expr.name.clone(), value
                    })))
                },
                _ => {
                    eprintln!("not an lvalue for var assignment");
                    self.error(&equals_tok, "not lvalue for var assignment");
                    Box::new(new_expr(ExprKind::ParseError))
                }
            }
        } else {
        // no '=' was parsed so it must be just a logical_or expression
            expr
        }
    }

    // logical_or := logic_and ( 'or' logical_and)*
    fn logical_or(&mut self) -> Box<Expr> {
        let mut expr = self.logical_and();
        while self.match_any_of(&[LoxToken![Or]]) {
           let op = self.previous().clone();
           let right_expr = self.logical_and(); 
           expr = Box::new(new_expr(ExprKind::Logical(LogicalExpr { left: expr, op, right: right_expr })));
        }
        expr
    }

    // logical_and := equality ( 'and' equality)*
    fn logical_and(&mut self) -> Box<Expr> {
       let mut expr = self.equality();
        while self.match_any_of(&[LoxToken![And]]) {
           let op = self.previous().clone();
           let right_expr = self.equality(); 
           expr = Box::new(new_expr(ExprKind::Logical(LogicalExpr { left: expr, op, right: right_expr })));
        }
        expr
    }

    // equality := comparison ( ( "!=" | "==" ) comparison )*
    fn equality(&mut self) -> Box<Expr> {
        let mut expr = self.comparison();
        while self.match_any_of(&[LoxToken![BangEqual], LoxToken![EqualEqual]]) {
            let op = self.previous().clone();
            let right_expr = self.comparison();
            expr = Box::new(new_expr(ExprKind::Binary(BinaryExpr{ left: expr, op, right: right_expr})));
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
            expr = Box::new(new_expr(ExprKind::Binary(BinaryExpr{ left: expr, op, right: right_expr})));
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
            expr = Box::new(new_expr(ExprKind::Binary(BinaryExpr{ left: expr, op, right: right_expr})));
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
            expr = Box::new(new_expr(ExprKind::Binary(BinaryExpr{ left: expr, op, right: right_expr})));
        }
        expr
    }

    // unary :=   ("!" | "-") unary 
    //          | call;
    fn unary(&mut self) -> Box<Expr> {
        if self.match_any_of(&[LoxToken![Bang], LoxToken![Minus]]) {
            let op = self.previous().clone();
            let expr = self.unary();
            Box::new(new_expr(ExprKind::Unary(UnaryExpr{ op, expr })))
        } else {
            return self.call();
        }
    }

    // call parses the following: either a primary, funcall() or obj.field 
    // call := primary ( "(" arguments? ")" | "." IDENTIFIER )* 
    fn call(&mut self) -> Box<Expr> {
        // this is interesting parsing trick, if there are no () suffix, its just a 
        // regular primary expression not a call expression
        let mut expr = self.primary();
        loop {
            // if we find a ( next then it must be a call
            if self.match_any_of(&[LoxToken![LeftParen]]) {
                expr = self.finish_call(expr);
            } else if self.match_any_of(&[LoxToken![Dot]]) {
                let name = self.consume(&LoxToken![Identifier("".to_string())], "expect property after '.'");
                match name {
                    Ok(tok) => {
                        expr = Box::new(new_expr(ExprKind::Get(GetExpr{ object: expr, name: tok.clone()})));
                    },
                    Err(_) => {
                        expr = Box::new(new_expr(ExprKind::ParseError));
                        return expr;
                    }
                }
            } else {
                break;
            }
        }
        expr
    }
    
    // gathers up arguments and packages it up into Expr::Call instance
    fn finish_call(&mut self, callee: Box<Expr>) -> Box<Expr> {
        let mut arguments = Vec::new();
        if !self.check(&LoxToken![RightParen]) {
            loop {
                arguments.push(self.expression());
                if arguments.len() > 255 {
                    let tok = self.peek().clone();
                    self.error(&tok, "can't have more than 255 arguments for function call");
                }
                if !self.match_any_of(&[LoxToken![Comma]]) {
                    break;
                }
            }
        }
        let paren_tok = self.consume(&LoxToken![RightParen], "expected ')' after arguments").unwrap();
        Box::new(new_expr(ExprKind::Call(CallExpr{ callee, paren: paren_tok.clone(), arguments})))
    }

    // primary := NUMBER | STRING | "true" | "false" | "nil" 
    //           | identifier 
    //           | '(' expression ')'
    fn primary(&mut self) -> Box<Expr> {
        let num_tok = TokenType::Number(0.0);
        let str_tok = TokenType::String(String::new());
        if self.match_any_of(&[num_tok, str_tok, LoxToken![True], LoxToken![False], LoxToken![Nil]]) {
            let expr = match self.previous().token_type {
                TokenType::Number(n) => 
                    ExprKind::Literal(LiteralExpr{ val: LoxValue::Number(n) }),
                TokenType::String(ref s) =>
                    ExprKind::Literal(LiteralExpr{ val: LoxValue::String(s.clone()) }),
                TokenType::True => 
                    ExprKind::Literal(LiteralExpr{ val: LoxValue::Bool(true) }),
                TokenType::False => 
                    ExprKind::Literal(LiteralExpr{ val: LoxValue::Bool(false) }),
                TokenType::Nil => 
                    ExprKind::Literal(LiteralExpr{ val: LoxValue::Nil}),
                    _ => { ExprKind::ParseError }
            };
            Box::new(new_expr(expr))
        } else if self.match_any_of(&[LoxToken![Identifier("".to_string())]]) {
            Box::new(new_expr(ExprKind::Variable(VariableExpr{name:self.previous().clone()})))
        } else if self.match_any_of(&[LoxToken![LeftParen]]) {
            let expr = self.expression();
            let result = self.consume(&LoxToken![RightParen], "expected ')' after expression");
            match result {
                Ok(_) => {
                    Box::new(new_expr(ExprKind::Grouping(GroupingExpr{ group: Box::new(*expr) })))
                },
                Err(_) => {
                    Box::new(new_expr(ExprKind::ParseError))
                }
            }
        } else {
            let tok = self.peek().to_owned();
            self.error(&tok, "parse primary fail");
            Box::new(new_expr(ExprKind::ParseError))
        }
    }

    fn error(&self, tok: &Token, msg: impl Into<String>) {
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

pub fn do_decl(source: impl Into<String>) -> Result<Box<Stmt>, ParseError> {
    let mut parser = Parser::new(&gen_tokens(source.into().as_str()));
    let stmt = parser.declaration();
    parser.check_parse_error_stmt(stmt)
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

    #[test]
    fn test_parse_function_expr() {
        let expr = do_decl("fun foo(a1, a2) { }")
            .map(|expr| expr.to_string())
            .unwrap_or_else(|_| {
                "error".to_string()
            });
        assert_ne!(expr, "error");
    }
}

