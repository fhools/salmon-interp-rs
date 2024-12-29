use super::lex;
trait ExprVisitor<R> {
    fn visit_expr(&mut self, expr: &Expr) -> R;
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
#[derive(Debug)]
pub struct AssignExpr {
    name: lex::Token,
    value: Box<Expr>,
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
pub struct GroupingExpr;
#[derive(Debug)]
pub struct LiteralExpr {
   pub val: LoxValue
}
#[derive(Debug)]
pub struct LogicalExpr;
#[derive(Debug)]
pub struct SetExpr;
#[derive(Debug)]
pub struct SuperExpr;
#[derive(Debug)]
pub struct ThisExpr;
#[derive(Debug)]
pub struct UnaryExpr {
    pub op: lex::Token,
    pub val: LoxValue
}
#[derive(Debug)]
pub struct VariableExpr;

pub struct PrintVisitor;
impl ExprVisitor<String> for PrintVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Binary(b) => {
                format!("{} {:?} {}", self.visit_expr(&b.left), b.op.lexeme , self.visit_expr(&b.right))
            },
            Expr::Call(_) => { String::new()},
            Expr::Get(_) => {String::new()},
            Expr::Grouping(_) => {String::new()},
            Expr::Literal(lit) => {lit.val.to_string()},
            Expr::Logical(LogicalExpr) => {String::new()},
            Expr::Set(SetExpr) => {String::new()},
            Expr::Super(SuperExpr) => {String::new()},
            Expr::This(ThisExpr) => {String::new()},
            Expr::Unary(UnaryExpr) => {String::new()},
            Expr::Variable(VariableExpr) => {String::new()},
        }

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
        eprintln!("print visitor: {}", output);
        assert!(!output.is_empty());
    }
}


