use super::lex;
use std::fmt::{self, Display};
use super::interp::RuntimeError;

#[derive(Debug)]
pub enum Stmt {
    Expression(Box<Expr>),
    Print(Box<Expr>),
    VarDecl(VarDecl),
    Block(Block),
    If(IfStmt),
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
        }
    }
}

#[derive(Debug)]
pub struct Block {
    pub statements: Vec<Stmt> 
}
#[derive(Debug)]
pub struct VarDecl {
    pub name: lex::Token,
    pub initializer: Option<Box<Expr>>
}

#[derive(Debug)]
pub struct IfStmt {
    pub conditional: Box<Expr>,
    pub then_branch: Box<Stmt>,
    pub else_branch: Option<Box<Stmt>>
}

#[derive(Debug)]
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

#[derive(Debug, Clone)]
pub enum LoxValue {
    Nil,
    Bool(bool),
    Number(f32),
    String(String)
}

impl ToString for LoxValue {
    fn to_string(&self) -> String {
        match self {
            LoxValue::Nil => "nil".to_string(),
            LoxValue::Bool(b) => format!("{}", b),
            LoxValue::Number(n) => format!("{}", n),
            LoxValue::String(s) => s.clone()
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = PrintVisitor;
        write!(f, "{}", printer.visit_expr(self))
    }
}

#[derive(Debug)]
pub struct AssignExpr {
    pub name: lex::Token,
    pub value: Box<Expr>,
}

#[derive(Debug)]
pub struct BinaryExpr {
    pub left: Box<Expr>,
    pub op: lex::Token,
    pub right: Box<Expr>,
}
#[derive(Debug)]
pub struct CallExpr;
#[derive(Debug)]
pub struct GetExpr;

#[derive(Debug)]
pub struct GroupingExpr {
    pub group: Box<Expr>
}

#[derive(Debug)]
pub struct LiteralExpr {
   pub val: LoxValue
}
#[derive(Debug)]
pub struct LogicalExpr {
    pub left: Box<Expr>,
    pub op: lex::Token,
    pub right: Box<Expr>
}

#[derive(Debug)]
pub struct SetExpr;
#[derive(Debug)]
pub struct SuperExpr;
#[derive(Debug)]
pub struct ThisExpr;
#[derive(Debug)]
pub struct UnaryExpr {
    pub op: lex::Token,
    pub expr: Box<Expr>
}
#[derive(Debug)]
pub struct VariableExpr {
    pub name: lex::Token
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
            Expr::Call(_) => { String::new()},
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


