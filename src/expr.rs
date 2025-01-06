use super::lex::{self, Token};
use std::fmt::{self, Display};
use std::io::Write;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    Return(ReturnStmt),
    Class(ClassStmt),
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
            Stmt::Return(return_stmt) => {
                match return_stmt.value {
                 Some(ref value_expr) => format!("(return {})", print_visitor.visit_expr(value_expr)),
                 None => format!("(return)") 
                }
            },
            Stmt::Class(class_stmt) => {
                let mut methods = String::new();
                for a in &class_stmt.methods {
                    methods = format!("{} {}", methods, a.name.lexeme);
                }
                format!("(class {} (methods {})", class_stmt.name.lexeme, methods)
            }

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


// Used to generate ids for Expr
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
pub fn generate_expr_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}
pub fn new_expr(exprkind: ExprKind) -> Expr {
    Expr { id: generate_expr_id(), kind: exprkind}
}
#[derive(Debug, Clone)]
pub struct Expr {
    pub id: usize,
    pub kind: ExprKind
}

#[derive(Debug, Clone)]
pub enum ExprKind {
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

    // Dont know yet if this is really part of Crafting Interpreter's design
    Function(Box<dyn LoxCallable>),
    // class
    Class(LoxClass),
    Instance(LoxInstance),

    // Note: this is not in Crafting Interpreters, its because Rust has no exceptions, 
    // to serve as an unwind mechanism for return statements
    Return(Option<Box<LoxValue>>)
}

pub trait LoxCallable : Debug {
    fn call(&self, interpreter: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError>;
    fn clone_box(&self) -> Box<dyn LoxCallable>;
}

#[derive(Debug)]
pub struct LoxFunction {
    // TODO: should I store this as a Box<Stmt> instead?
    pub function: Box<FunctionStmt>,
    pub closure: usize,
}

impl LoxCallable for LoxFunction {
    fn call(&self, interp: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError> {
        //let global_env = interp.global_env_id();
        let call_env = interp.push_env(self.closure);
        for (i,arg) in arguments.iter().enumerate() {
            interp.define(&self.function.params.get(i).unwrap().lexeme, call_env, arg.clone());
        }
        let return_val = if let Ok(LoxValue::Return(Some(call_value))) = interp.execute_block(&self.function.body, call_env) {
            (*call_value).clone()
        }  else  {
            LoxValue::Nil
        };
        // FIXME: buggy, a call may return closure, maybe we should never even pop env
        interp.pop_env();
        Ok(return_val)
    }

    fn clone_box(&self) -> Box<dyn LoxCallable> {
        // TODO: implement LoxFunction Clone
        Box::new(LoxFunction{function: self.function.clone(), closure: self.closure})
    }
}

#[derive(Debug, Clone)]
pub struct LoxClass {
    pub name: Token,
}

#[derive(Debug, Clone)]
pub struct LoxInstance {
    pub klass: Box<LoxClass>,
    pub fields: HashMap<String, LoxValue>,
}

impl LoxInstance {
    pub fn get(&self, name: &Token) -> Result<LoxValue, RuntimeError> {
        Ok(LoxValue::Number(100.0))
        //Err(RuntimeError::General("lox instance get unimplemented"))
    }
}
impl LoxCallable for LoxClass {
    fn call(&self, interp: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError> {
        eprintln!("calling class constructor for class {}", self.name.lexeme);
        let lox_instance = 
            LoxValue::Instance(LoxInstance{
                klass: Box::new(self.clone()),
                fields: HashMap::default(),
            });
        Ok(lox_instance)
    }

    fn clone_box(&self) -> Box<dyn LoxCallable> {
        Box::new(LoxClass{name: self.name.clone()})
    }
}

impl LoxCallable for LoxInstance {
    fn call(&self, interp: &mut Interpreter, arguments: &Vec<LoxValue>) -> Result<LoxValue, RuntimeError> {
        todo!("called call() on LoxInstance");
        Ok(LoxValue::Nil)
    }
    fn clone_box(&self) -> Box<dyn LoxCallable> {
        Box::new(LoxInstance{
            klass: self.klass.clone(),
            fields: self.fields.clone(),
        })
    }
}

impl ToString for LoxValue {
    fn to_string(&self) -> String {
        match self {
            LoxValue::Nil => "nil".to_string(),
            LoxValue::Bool(b) => format!("{}", b),
            LoxValue::Number(n) => format!("{}", n),
            LoxValue::String(s) => s.clone(),
            LoxValue::Function(_l) => format!("callable()"),
            LoxValue::Return(None) =>  format!("return"),
            LoxValue::Return(Some(ref box_lv)) => format!("{}",box_lv.to_string()),
            LoxValue::Class(c) => format!("class {}", c.name.lexeme),
            LoxValue::Instance(i) => format!("instance of {}", i.klass.name.lexeme),
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
pub struct GetExpr{
    pub object: Box<Expr>,
    pub name: Token, 
}

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

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub return_tok: Token,
    pub value: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct ClassStmt {
    pub name: Token,
    pub methods: Vec<FunctionStmt> 
}

pub trait ExprVisitor<R> {
    fn visit_expr(&mut self, expr: &Expr) -> R;
    fn visit_logical_expr(&mut self, expr: &Expr) -> R;
}

impl ExprVisitor<String> for PrintVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match &expr.kind{
            ExprKind::Binary(b) => {
                format!("({} {} {})",
                        b.op.lexeme,
                        self.visit_expr(&b.left),
                        self.visit_expr(&b.right))
            },
            ExprKind::Call(call) => { 
                let mut arguments = String::new();
                for a in &call.arguments {
                    arguments = format!("{}{}", &arguments, self.visit_expr(&a));
                }
                format!("(call {} (args: {}))", 
                        self.visit_expr(&call.callee),
                        arguments)
            },
            ExprKind::Get(_) => {String::new()},
            ExprKind::Grouping(_) => {String::new()},
            ExprKind::Literal(lit) => {lit.val.to_string()},
            ExprKind::Logical(logical_expr) => {
                self.visit_logical_expr(expr)
            }
            ExprKind::Set(SetExpr) => {String::new()},
            ExprKind::Super(SuperExpr) => {String::new()},
            ExprKind::This(ThisExpr) => {String::new()},
            ExprKind::Unary(unary) => {
                format!("({} {})", 
                        unary.op.lexeme,
                        self.visit_expr(&unary.expr))
            },
            ExprKind::Variable(var_expr) => {
                // TODO: look up variable in environment and visit the expression value
                format!("{}", var_expr.name.lexeme)
            },
            ExprKind::Assign(assign_expr) => {
                format!("(assign {} {})", assign_expr.name.lexeme, self.visit_expr(&assign_expr.value))
            },
            ExprKind::ParseError => "parse_error".to_string()
        }

    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Logical(l) => {
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
        match &expr.kind {
            ExprKind::Binary(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            },
            ExprKind::Call(_) => { },
            ExprKind::Get(_) => {},
            ExprKind::Grouping(grp) => {
                self.visit_expr(&grp.group);
            },
            ExprKind::Literal(_lit) => {},
            ExprKind::Logical(logical_expr) => {
                self.visit_logical_expr(expr);
            },
            ExprKind::Set(SetExpr) => {},
            ExprKind::Super(SuperExpr) => {},
            ExprKind::This(ThisExpr) => {},
            ExprKind::Unary(_unary) => {},
            ExprKind::Variable(VariableExpr) => {},
            ExprKind::Assign(_) => {},
            ExprKind::ParseError =>  {
                self.has_error = true;
            }
        }
    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> () {
        match &expr.kind {
            ExprKind::Logical(logical_expr) => {
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
        let bexpr = Expr{ 
            id: generate_expr_id(),
            kind: ExprKind::Binary(BinaryExpr{ 
                left: Box::new(Expr{ id: generate_expr_id(),
                    kind: ExprKind::Literal(LiteralExpr{val: lox_num(1.0)
                    })}),
                op: lex::Token::new(TokenType::Plus, "+".to_string(), 0),
               right: Box::new(Expr { id: generate_expr_id(), 
                        kind: ExprKind::Literal(LiteralExpr{val: lox_num(2.0)})})
                                  })};
                    
        let mut visitor =  PrintVisitor{};
        let output = visitor.visit_expr(&bexpr);
        assert!(!output.is_empty());
    }
}


