use super::lex::{self, Token};
use std::fmt::{self, Display};
use std::io::Write;
use std::fmt::Debug;
use super::interp::{RuntimeError, Interpreter};

#[derive(Debug, Clone)]
pub enum Stmt {
    Expression(Box<Expr>),
    Print(Box<Expr>),
    VarDecl(VarDecl),
    Block(Block),
    If(IfStmt),
    While(WhileStmt),
    Function(FunctionStmt),
    ParseError,
}

pub trait StmtVisitor<R> {
    fn visit_stmt(&mut self, stmt: &Stmt) -> R;
}

pub struct PrintVisitor;
impl StmtVisitor<String> for PrintVisitor {
    fn visit_stmt(&mut self, stmt: &Stmt) -> String {
        let mut print_visitor = PrintVisitor;
        match stmt {
            Stmt::Expression(ref expr) => {
               print_visitor.visit_expr(&*expr)
            },
            Stmt::Print(expr) => {
                format!("(print {})", print_visitor.visit_expr(expr))
            },
            Stmt::VarDecl(var_decl) =>  {
                format!("(var_decl {} {})", var_decl.name.lexeme, 
                        var_decl.initializer
                        .as_ref()
                        .map_or("null".to_string(), |expr| print_visitor.visit_expr(expr)))
            },
            Stmt::Block(block) => {
                let mut output;
                output = format!("{{");
                for (i, v) in block.statements.iter().enumerate() {
                    if i > 0 {
                        output = format!("{},", output);
                    }
                    output += &format!("{}", self.visit_stmt(v));
                }
                output
            },
            Stmt::If(if_stmt) => {
                format!("(if {} {} {})",
                print_visitor.visit_expr(&if_stmt.conditional),
                print_visitor.visit_stmt(&if_stmt.then_branch),
                print_visitor.visit_stmt(&if_stmt.then_branch))

            }
            Stmt::While(while_stmt) => { 
                format!("(while {} {})",
                print_visitor.visit_expr(&while_stmt.condition),
                print_visitor.visit_stmt(&while_stmt.body))
            },
            Stmt::Function(function) => {
                let mut params = String::new();
                for a in &function.params {
                    params = format!("{} {}",params, a.lexeme);
                }
                format!("(func_decl {} {} ...) ", function.name.lexeme, params)
            },
            Stmt::ParseError => { "Stmt::ParseError".to_string()},
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Stmt> 
}
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: Token,
    pub initializer: Option<Box<Expr>>
}

#[derive(Debug, Clone)]
pub enum Expr {
    Binary(BinaryExpr),
    Call(CallExpr),
    Get(GetExpr),
    Grouping(GroupingExpr),
    Literal(LiteralExpr),
    Logical(LogicalExpr),
    Set(SetExpr),
    Super(SuperExpr),
    This(ThisExpr),
    Unary(UnaryExpr),
    Variable(VariableExpr),
    Assign(AssignExpr),
    ParseError
}
impl Clone for Box<dyn LoxCallable> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
#[derive(Debug, Clone)]
pub enum LoxValue {
    Nil,
    Bool(bool),
    Number(f32),
    String(String),
    Function(Box<dyn LoxCallable>)
}

pub trait LoxCallable : Debug {
    fn call(&self, interpreter: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError>;
    fn clone_box(&self) -> Box<dyn LoxCallable>;
}

#[derive(Debug)]
pub struct LoxFunction {
    // TODO: should I store this as a Box<Stmt> instead?
    pub function: Box<FunctionStmt>,
}

impl LoxCallable for LoxFunction {
    fn call(&self, interp: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError> {
        let global_env = interp.global_env_id();
        let call_env = interp.push_env(global_env);
        for (i,arg) in arguments.iter().enumerate() {
            interp.define(&self.function.params.get(i).unwrap().lexeme, call_env, arg.clone());
        }
        interp.execute_block(&self.function.body, call_env);
        // FIXME: buggy, a call may return closure, maybe we should never even pop env
        interp.pop_env();
        Ok(LoxValue::Nil)
    }

    fn clone_box(&self) -> Box<dyn LoxCallable> {
        Box::new(LoxFunction{function: self.function.clone()})
    }
}

impl ToString for LoxValue {
    fn to_string(&self) -> String {
        match self {
            LoxValue::Nil => "nil".to_string(),
            LoxValue::Bool(b) => format!("{}", b),
            LoxValue::Number(n) => format!("{}", n),
            LoxValue::String(s) => s.clone(),
            LoxValue::Function(_l) => format!("callable()")
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = PrintVisitor;
        write!(f, "{}", printer.visit_expr(self))
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = PrintVisitor;
        write!(f, "{}", printer.visit_stmt(self))
    }
}

#[derive(Debug, Clone)]
pub struct AssignExpr {
    pub name: lex::Token,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expr>,
    pub op: lex::Token,
    pub right: Box<Expr>,
}
#[derive(Debug, Clone)]
pub struct CallExpr {
    pub callee: Box<Expr>,
    pub paren: lex::Token,
    pub arguments: Vec<Box<Expr>>,
}
#[derive(Debug, Clone)]
pub struct GetExpr;

#[derive(Debug, Clone)]
pub struct GroupingExpr {
    pub group: Box<Expr>
}

#[derive(Debug, Clone)]
pub struct LiteralExpr {
   pub val: LoxValue
}
#[derive(Debug, Clone)]
pub struct LogicalExpr {
    pub left: Box<Expr>,
    pub op: lex::Token,
    pub right: Box<Expr>
}

#[derive(Debug, Clone)]
pub struct SetExpr;
#[derive(Debug, Clone)]
pub struct SuperExpr;
#[derive(Debug, Clone)]
pub struct ThisExpr;
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: lex::Token,
    pub expr: Box<Expr>
}
#[derive(Debug, Clone)]
pub struct VariableExpr {
    pub name: Token
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub conditional: Box<Expr>,
    pub then_branch: Box<Stmt>,
    pub else_branch: Option<Box<Stmt>>
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Box<Expr>,
    pub body: Box<Stmt>,
}

#[derive(Debug, Clone)]
pub struct FunctionStmt {
    pub name: Token,
    pub params: Vec<Token>,
    pub body: Vec<Stmt>,
}

pub trait ExprVisitor<R> {
    fn visit_expr(&mut self, expr: &Expr) -> R;
    fn visit_logical_expr(&mut self, expr: &Expr) -> R;
}

impl ExprVisitor<String> for PrintVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Binary(b) => {
                format!("({} {} {})",
                        b.op.lexeme,
                        self.visit_expr(&b.left),
                        self.visit_expr(&b.right))
            },
            Expr::Call(call) => { 
                let mut arguments = String::new();
                for a in &call.arguments {
                    arguments = format!("{}{}", &arguments, self.visit_expr(&a));
                }
                format!("(call {} (args: {}))", 
                        self.visit_expr(&call.callee),
                        arguments)
            },
            Expr::Get(_) => {String::new()},
            Expr::Grouping(_) => {String::new()},
            Expr::Literal(lit) => {lit.val.to_string()},
            Expr::Logical(logical_expr) => {
                self.visit_logical_expr(expr)
            }
            Expr::Set(SetExpr) => {String::new()},
            Expr::Super(SuperExpr) => {String::new()},
            Expr::This(ThisExpr) => {String::new()},
            Expr::Unary(unary) => {
                format!("({} {})", 
                        unary.op.lexeme,
                        self.visit_expr(&unary.expr))
            },
            Expr::Variable(var_expr) => {
                // TODO: look up variable in environment and visit the expression value
                format!("{}", var_expr.name.lexeme)
            },
            Expr::Assign(assign_expr) => {
                format!("(assign {} {})", assign_expr.name.lexeme, self.visit_expr(&assign_expr.value))
            },
            Expr::ParseError => "parse_error".to_string()
        }

    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Logical(l) => {
                format!("({} {} {})",
                l.op.lexeme,
                self.visit_expr(&l.left),
                self.visit_expr(&l.right))
            },
            _ => {
                "parse_error".to_string()
            }

        }
    }

}

pub struct ErrorVisitor {
    has_error: bool
}
impl StmtVisitor<()> for ErrorVisitor {
    fn visit_stmt(&mut self, stmt: &Stmt) {
    }
}
impl ExprVisitor<()> for ErrorVisitor {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Binary(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            },
            Expr::Call(_) => { },
            Expr::Get(_) => {},
            Expr::Grouping(grp) => {
                self.visit_expr(&grp.group);
            },
            Expr::Literal(_lit) => {},
            Expr::Logical(logical_expr) => {
                self.visit_logical_expr(expr);
            },
            Expr::Set(SetExpr) => {},
            Expr::Super(SuperExpr) => {},
            Expr::This(ThisExpr) => {},
            Expr::Unary(_unary) => {},
            Expr::Variable(VariableExpr) => {},
            Expr::Assign(_) => {},
            Expr::ParseError =>  {
                self.has_error = true;
            }
        }
    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> () {
        match expr {
            Expr::Logical(logical_expr) => {
                self.visit_expr(&logical_expr.left);
                self.visit_expr(&logical_expr.right);
            },
            _ => {}
        }
    }
}

impl ErrorVisitor {
    pub fn new() -> Self {
        ErrorVisitor {
            has_error: false
        }
    }
    pub fn has_error_node(&self) -> bool {
        self.has_error
    }
}
mod test {
    use super::*;
    use super::lex::{TokenType};

    fn lox_num(n: f32) -> LoxValue {
        LoxValue::Number(n)
    }

    #[test]
    fn print_expr() {
        let bexpr = Expr::Binary(BinaryExpr{ 
            left: Box::new(Expr::Literal(LiteralExpr{val: lox_num(1.0)})),
            op: lex::Token::new(TokenType::Plus, "+".to_string(), 0),
            right: Box::new(Expr::Literal(LiteralExpr{val: lox_num(2.0)})),
        });
        let mut visitor =  PrintVisitor{};
        let output = visitor.visit_expr(&bexpr);
        assert!(!output.is_empty());
    }
}


