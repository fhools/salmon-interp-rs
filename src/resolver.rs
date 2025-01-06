use crate::interp::Interpreter;
use crate::lex::Token;
use crate::salmon_error;
use crate::expr::{ExprKind, Expr, Stmt, Block, FunctionStmt, ReturnStmt, WhileStmt, BinaryExpr, CallExpr, GroupingExpr, LogicalExpr, UnaryExpr, ClassStmt, GetExpr, SetExpr};
use std::collections::HashMap;

/* 
 * The Resolver does a post-parsing analysis on the AST. It essential keeps track of the scope
 * depth of every variable, and also keeps track of the scope depth for every expression so that
 * when a variable is accessed the expression will utilize the same scope depth during runtime, 
 * that was tracking during the resolution analysis. 
 *
 * Essentially this is implementing lexical scoping rules. At least this is my understanding of
 * what Chapter 11 was about!
 */
pub struct Resolver {
    scopes: Vec<HashMap<String, bool>>,
}

impl Default for Resolver {
    fn default() -> Self {
        Resolver {
            scopes: Vec::new()
        }
    } 
}
impl Resolver {
    pub fn resolve(&mut self, interp: &mut Interpreter, stmts: &Vec<Stmt>) {
        for s in stmts {
            self.resolve_stmt(interp, s);
        }
    }

    fn resolve_stmt(&mut self, interp: &mut Interpreter, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl(ref var_decl) => {
                //eprintln!("variable decl {}", var_decl.name.lexeme);
                self.declare(&var_decl.name);
                if let Some(expr) = &var_decl.initializer {
                    self.resolve_expr(interp, expr);
                }
                self.define(&var_decl.name);
            },
            Stmt::Function(ref function_stmt) => {
                self.declare(&function_stmt.name);
                self.define(&function_stmt.name);
                self.resolve_function(interp, function_stmt);
            },

            Stmt::Expression(ref expr) => {
                self.resolve_expr(interp, expr);
            },

            Stmt::If(ref if_stmt) => {
                self.resolve_expr(interp, &if_stmt.conditional);
                self.resolve_stmt(interp, &if_stmt.then_branch);
                if let Some(ref else_branch) = if_stmt.else_branch {
                    self.resolve_stmt(interp, else_branch);
                }
            },

            Stmt::Print(ref expr) => {
                self.resolve_expr(interp, expr);
            },

            Stmt::Return(ReturnStmt{ value: Some(ref expr), ..}) => {
                    self.resolve_expr(interp, expr);
            },

            Stmt::While(WhileStmt{ ref condition, ref body }) => {
                self.resolve_expr(interp, condition);
                self.resolve_stmt(interp, body);
            },
            Stmt::Block(Block{ ref statements }) => {
                self.begin_scope();
                self.resolve(interp, statements);
                self.end_scope();
            },
            Stmt::Class(ClassStmt{ ref name, ref methods }) => {
                //eprintln!("resolver defining {}", name.lexeme);
                self.declare(name);
                self.define(name);
                self.begin_scope();
                if let Some(last) = self.scopes.last_mut() {
                    last.insert("this".to_string(), true);
                } else {
                    eprintln!("resolve couldn't insert this keywotrd");
                }
                for method in methods {
                    self.resolve_function(interp, method);
                }
                self.end_scope()

            },
            _ => {}
        }
    }

    fn resolve_expr(&mut self, interp: &mut Interpreter, expr: &Expr) {
        match &expr.kind {
            ExprKind::Variable(var) => {
                if !self.scopes.is_empty() {
                    if let Some(last) = self.scopes.last() {
                        if last.get(&var.name.lexeme).is_some_and(|v| !v) {
                            salmon_error(var.name.line, "can't read local variable in its own initializer");
                            panic!("can't read local variable in its own initializer");
                        }
                    }
                }
                self.resolve_local(interp, expr, &var.name);
            },

            ExprKind::Assign(ref assign_expr) => {
                self.resolve_expr(interp, &assign_expr.value);
                self.resolve_local(interp, expr, &assign_expr.name);
            },

            ExprKind::Binary(BinaryExpr{ ref left, ref right, ..}) => {
                self.resolve_expr(interp, left);
                self.resolve_expr(interp, right);
            },
            ExprKind::Call(CallExpr{ref callee, ref arguments, ..}) => {
                self.resolve_expr(interp, callee);
                for arg in arguments {
                    self.resolve_expr(interp, arg);
                }
            },
            ExprKind::Grouping(GroupingExpr{ ref group }) => {
                self.resolve_expr(interp, group);
            },
            ExprKind::Logical(LogicalExpr{ ref left, ref right, ..}) => {
                self.resolve_expr(interp, left);
                self.resolve_expr(interp, right);
            },
            ExprKind::Unary(UnaryExpr{ ref expr, ..}) => {
                self.resolve_expr(interp, expr);
            },
            ExprKind::Get(GetExpr{ref object, ..}) => {
                self.resolve_expr(interp, object);
            },
            ExprKind::Set(SetExpr{ref object, ref value, ..}) => {
                self.resolve_expr(interp, object);
                self.resolve_expr(interp, value);
            },
            ExprKind::This(ref this_expr) => {
                //eprintln!("resolving this the tok is: {}, scope len is {}", this_expr.keyword, self.scopes.len());
                self.resolve_local(interp, expr, &this_expr.keyword);
            },
           _ => {} 
        }
    }

    fn resolve_local(&mut self, interp: &mut Interpreter, expr: &Expr, name_tok: &Token) {
        // TODO: fix this code to not have janky  test is empty
        if self.scopes.is_empty() {
            //eprintln!("resolve local resolve to global variable: {}", name_tok.lexeme);
            return;
        } 
        let mut i = (self.scopes.len() - 1) as isize; 

        // This loop finds the variables scope depth, searching from the inner most scope to the outer
        // most scope. When it finds it it registers with the interpreter the expression,
        // the depth it used.
        while i >= 0  {
            //eprintln!("resolve local var {} depth: {}", name_tok.lexeme, i);
            if self.scopes.get_mut(i as usize).unwrap().contains_key(&name_tok.lexeme) {
                //eprintln!("resolving expr id: {} of var: {} to depth of {}", expr.id, name_tok.lexeme, self.scopes.len() - 1 - (i as usize));
                interp.resolve(expr, self.scopes.len() - 1  - (i as usize));
                return;
            }
            i -= 1; 
        }
    }

    fn resolve_function(&mut self, interp: &mut Interpreter, function_stmt: &FunctionStmt) {
        self.begin_scope();
        for p in  &function_stmt.params {
            self.declare(p);
            self.define(p);
        }
        self.resolve(interp, &function_stmt.body);
        self.end_scope();

    }
    fn declare(&mut self, tok: &Token) {
        if self.scopes.is_empty() {
            return;
        }
        if let Some(last) = self.scopes.last_mut() {
            //eprintln!("declare {}", tok.lexeme);
            last.insert(tok.lexeme.to_owned(), false);
        }
    }

    fn define(&mut self, tok: &Token) {
        if self.scopes.is_empty() {
            return;
        }
        if let Some(last) = self.scopes.last_mut() {
            //eprintln!("define {}", tok.lexeme);
            last.insert(tok.lexeme.to_owned(), true);
        }
    }

    fn begin_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn end_scope(&mut self) {
        self.scopes.pop();
    }

}
