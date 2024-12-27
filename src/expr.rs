use super::lex;
trait ExprVisitor {
    fn visit_assignexpr(&mut self, expr: &AssignExpr) -> bool;
    fn visit_binaryexpr(&mut self, expr: &BinaryExpr) -> bool;
    fn visit_callexpr(&mut self, expr: &CallExpr) -> bool;
    fn visit_getexpr(&mut self, expr: &GetExpr) -> bool;
    fn visit_groupingexpr(&mut self, expr: &GroupingExpr) -> bool;
    fn visit_literalexpr(&mut self, expr: &LiteralExpr) -> bool;
    fn visit_logicalexpr(&mut self, expr: &LogicalExpr) -> bool;
    fn visit_setexpr(&mut self, expr: &SetExpr) -> bool;
    fn visit_superexpr(&mut self, expr: &SuperExpr) -> bool;
    fn visit_thisexpr(&mut self, expr: &ThisExpr) -> bool;
    fn visit_unaryexpr(&mut self, expr: &UnaryExpr) -> bool;
    fn visit_variableexpr(&mut self, expr: &VariableExpr) -> bool;
}

trait ExprVisitable {
    fn accept(&self, v: &Box<dyn ExprVisitor>);
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
