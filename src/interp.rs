use crate::expr::*;
use crate::lex::TokenType;
pub struct Interpreter {
}

impl Interpreter {
    fn evaluate(&mut self, expr: &Expr) -> Result<LoxValue, ()> {
        self.visit_expr(expr)
    }
}

fn to_op_fn(op_tok: TokenType) -> Box<dyn Fn(&LoxValue, &LoxValue) -> Result<LoxValue, ()>> {
    Box::new(move |l, r| {
        let (lval, rval) = match (l, r) {
            (LoxValue::Number(ref l_val), LoxValue::Number(ref r_val)) => {
                (*l_val, *r_val)
            },
            _ => { (0.0, 0.0)}
        };
        eprintln!("op_fn l and r: {}, {}", lval, rval);
        match op_tok {
            TokenType::Plus => {
               // TODO: type check both numbers or strings 
               Ok(LoxValue::Number(lval + rval))
            }
            _ => {
                Ok(LoxValue::Number(f32::NAN))
            }
        }
    })
}

impl ExprVisitor<Result<LoxValue, ()>> for Interpreter {
    fn visit_expr(&mut self, expr: &Expr) -> Result<LoxValue, ()> {
        match expr {
            Expr::Binary(b) => {
                let op_fn = to_op_fn(b.op.token_type.clone());
                let lvalue = self.visit_expr(&b.left);
                let rvalue = self.visit_expr(&b.right);
                eprintln!("visit_expr binary: {:?}, {:?}", lvalue, rvalue);
                match (lvalue, rvalue) {
                    (Ok(ref lv), Ok(ref rv)) => {
                        op_fn(lv, rv)
                    },
                    _ => { Err(()) }
                }
            },
            Expr::Literal(lit) => {
                Ok(lit.val.clone())
            },
            _ => {
                Err(())
            }
        }
    }
}

fn evaluate_expr(expr: &Expr) -> LoxValue {
    let mut interpreter = Interpreter{};
    interpreter.evaluate(expr).unwrap_or(LoxValue::Nil)

}
mod test {
    use super::*;
    use super::super::parser::do_expr;

    #[test]
    fn test_evaluate_expr() {
        let expr_val = do_expr("1+2+3")
            .map(|expr| { 
                eprintln!("expr: {}", expr.to_string());
                evaluate_expr(&expr)
            })
            .map(|lvalue| {
                eprintln!("loxval: {:?}", lvalue);
                let val: f32 = match lvalue { LoxValue::Number(v) => v, _ => f32::NAN };
                val
            })
        .unwrap_or(102.0);
        eprintln!("evaluated expr to: {}", expr_val);
    }
}
