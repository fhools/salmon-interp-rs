use super::lex;
trait ExprVisitor<R> {
    fn visit_expr(&mut self, expr: &Expr) -> R;
}

#[derive(Debug)]
enum Expr {
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

#[derive(Debug)]
struct AssignExpr {
    name: lex::Token,
    value: Box<Expr>,
}

#[derive(Debug)]
struct BinaryExpr {
    left: Box<Expr>,
    op: lex::Token,
    right: Box<Expr>,
}
#[derive(Debug)]
struct CallExpr;
#[derive(Debug)]
struct GetExpr;
#[derive(Debug)]
struct GroupingExpr;
#[derive(Debug)]
struct LiteralExpr {
   val: String 
}
#[derive(Debug)]
struct LogicalExpr;
#[derive(Debug)]
struct SetExpr;
#[derive(Debug)]
struct SuperExpr;
#[derive(Debug)]
struct ThisExpr;
#[derive(Debug)]
struct UnaryExpr;
#[derive(Debug)]
struct VariableExpr;

struct PrintVisitor;
impl ExprVisitor<String> for PrintVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Binary(b) => {
                format!("{} {:?} {}", self.visit_expr(&b.left), b.op.lexeme , self.visit_expr(&b.right))
            },
            Expr::Call(_) => { String::new()},
            Expr::Get(_) => {String::new()},
            Expr::Grouping(_) => {String::new()},
            Expr::Literal(lit) => {lit.val.clone()},
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

    #[test]
    fn print_expr() {
        let bexpr = Expr::Binary(BinaryExpr{ 
            left: Box::new(Expr::Literal(LiteralExpr{val: "1".to_string()})),
            op: lex::Token::new(TokenType::Plus, "+".to_string(), 0),
            right: Box::new(Expr::Literal(LiteralExpr{val: "2".to_string()})),
        });
        let mut visitor =  PrintVisitor{};
        let output = visitor.visit_expr(&bexpr);
        eprintln!("print visitor: {}", output);
        assert!(!output.is_empty());
    }
}


