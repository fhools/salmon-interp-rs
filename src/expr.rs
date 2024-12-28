use super::lex;
trait ExprVisitor<R> {
    fn visit_expr(&mut self, expr: &Expr) -> R;
}

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

struct AssignExpr {
    name: lex::Token,
    value: Box<Expr>,
}

struct BinaryExpr {
    left: Box<Expr>,
    op: lex::Token,
    right: Box<Expr>,
}
struct CallExpr;
struct GetExpr;
struct GroupingExpr;
struct LiteralExpr;
struct LogicalExpr;
struct SetExpr;
struct SuperExpr;
struct ThisExpr;
struct UnaryExpr;
struct VariableExpr;

struct PrintVisitor();

//impl ExprVisitor<()> for PrintVisitor {
//    fn visit_expr(&mut self, expr: &Expr) {
//        match expr {
//            Expr::Binary(b) => {
//                println!("{} {} {}", left, op , right)
//            },
//            Expr::Call(_) => {},
//            Expr::Get(_) => {},
//        }
//
//    }
//
//}



