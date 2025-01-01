use crate::expr::*;
use std::collections::HashMap;
use crate::lex::TokenType;
pub struct Interpreter {
    pub top_level_env: Box<Environment>
}

#[derive(Default, Debug)]
pub struct Environment {
    pub enclosing_env: Option<Box<Environment>>,
    pub values: HashMap<String, LoxValue>,
}

impl Environment {
    fn get(&self, source: impl AsRef<str>) -> Result<LoxValue, RuntimeError> {
        let source: &str  = source.as_ref();
        self.values.get(source)
            .map_or_else(||  {
                Err(RuntimeError::General(Box::leak(format!("unknown variable {}", source).into_boxed_str())))
            }, |val| Ok(val.to_owned()))
    }

    fn set(&mut self, name: impl AsRef<str>, value: LoxValue) -> Result<(), RuntimeError> {
        self.values.insert(name.as_ref().to_owned(), value);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeError {
        General(&'static str)
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            top_level_env: Box::new(Environment::default()),
        }
    }

    fn evaluate(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        self.visit_expr(expr)
    }

    pub fn interpret(&mut self, statements: &Vec<Stmt>) -> Result<(), RuntimeError> {
        for stmt in statements {
            let result =  self.execute(stmt)?;
        }
        Ok(())
    }

    fn execute(&mut self, stmt: &Stmt) -> Result<(), RuntimeError> {
       match stmt {
           Stmt::Expression(expr) => {
                self.evaluate(expr).map_or_else(|e| Err(e), |v| Ok(()))
           },
           Stmt::Print(expr) => {
               let loxval = self.evaluate(expr);
               match loxval {
                   Ok(loxval) => {
                       println!("{}", loxval.to_string());
                       Ok(())
                   },
                   Err(err) => {
                       println!("runtime error: {:?}", err);
                       Err(err)
                   }
               }
           }
           Stmt::VarDecl(var_decl) => {
               let name = var_decl.name.lexeme.clone();
               // TODO: this is masking RuntimeError, what if the visit_expr fails return  valuue
               // that should bea runtime error instead of LoxNil
               let lox_value = var_decl
                   .initializer
                   .as_ref()
                   // if initializer is None then unwrap_or will return LoxValue nil
                   // otherwise map returns Option<Result<LoxValue...>>
                   .map(|expr| self.visit_expr(expr))
                   // ? returns Err of Result
                   .transpose()?
                   // unwrap will return LoxValue or nil if Option None
                   .unwrap_or(LoxValue::Nil);
                eprintln!("setting var: {}", name);
                self.top_level_env.set(name, lox_value);
                Ok(())   
           },
       }
    }
}

fn to_op_fn(op_tok: TokenType) -> Box<dyn Fn(&LoxValue, &LoxValue) -> Result<LoxValue, RuntimeError>> {
    Box::new(move |l, r| {
        match (&op_tok, l, r) {
            (TokenType::Plus, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval + rval))
            },
            (TokenType::Plus, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::String(format!("{}{}",lval, rval)))
            },
            (TokenType::Plus, _, _) => {
                Err(RuntimeError::General("+ operands must be both numbers or strings only"))
            },
            (TokenType::Minus, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval - rval))
            },
            (TokenType::Minus, _, _) => {
                Err(RuntimeError::General("- operands must be both numbers"))
            },
            // multiply
            (TokenType::Star, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval * rval))
            },
            (TokenType::Star, _, _) => {
                Err(RuntimeError::General("* operands must be both numbers"))
            },
            // divide
            (TokenType::Slash, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                if *rval == 0.0 {
                    Err(RuntimeError::General("divide by zero"))
                } else {
                    Ok(LoxValue::Number(lval / rval))
                }
            },
            (TokenType::Slash, _, _) => {
                Err(RuntimeError::General("/ operands must be both numbers"))
            },

            // TODO: handle logical comparison expressions! <=, >= , == , != , < , > 
            //
            _ => {
                Err(RuntimeError::General("binary expression unknown operator")) 
            }
        }
    })
}

fn is_truthy(loxval: &LoxValue) -> bool {
    match loxval {
        LoxValue::Nil => false,
        LoxValue::Bool(ref b) => *b,
        LoxValue::Number(_) | LoxValue::String(_) => true
    }
}

impl ExprVisitor<Result<LoxValue, RuntimeError>> for Interpreter {
    fn visit_expr(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        match expr {
            Expr::Binary(b) => {
                let op_fn = to_op_fn(b.op.token_type.clone());
                let lvalue = self.visit_expr(&b.left);
                let rvalue = self.visit_expr(&b.right);
                //eprintln!("visit_expr binary: {:?}, {:?}", lvalue, rvalue);
                match (lvalue, rvalue) {
                    (Ok(ref lv), Ok(ref rv)) => {
                        op_fn(lv, rv)
                    },
                    _ => { Err(RuntimeError::General("binary op failed on values")) }
                }
            },
            Expr::Literal(lit) => {
                Ok(lit.val.clone())
            },
            Expr::Unary(UnaryExpr{ op, ref expr }) => {
                let loxval = self.visit_expr(expr);
                //eprintln!("visit_expr unary op: {:?} inner expr: {:?}", op, loxval);
                match (&op.token_type, loxval) {
                    (TokenType::Minus, Ok(LoxValue::Number(value))) => {
                        Ok(LoxValue::Number(-value))
                    },
                    (TokenType::Bang, Ok(ref loxval)) => {
                        Ok(LoxValue::Bool(!is_truthy(loxval)))
                    },
                    _ => {
                        Err(RuntimeError::General("unary error"))
                    }
                }
            },
            Expr::Variable(VariableExpr{ ref name }) => {
                self.top_level_env.get(&name.lexeme)
                    .map_err(|_err| {
                    eprintln!("unfound variable: {}",name);
                    RuntimeError::General(Box::leak(format!("undefined variable: {}",name).into_boxed_str()))
                    })
            }
            a @ _ => {
                eprintln!("unhandled expr: {:?}", a);
                Err(RuntimeError::General("unhandled expr"))
            }
        }
    }
}

fn evaluate_expr(expr: &Expr) -> LoxValue {
    let mut interpreter = Interpreter::new();
    interpreter.evaluate(expr).unwrap_or_else(|RuntimeError::General(s)| {
        eprintln!("runtime error to evaluate: \"{}\" : {}", expr, s);
        LoxValue::Nil
    })

}
mod test {
    use super::*;
    use super::super::parser::{Parser, do_expr};
    use super::super::lex::gen_tokens;

    struct DoIt {
    }

    impl DoIt {
        fn interpret(&mut self, source: &str) -> Result<(), RuntimeError> {
            let mut parser = Parser::new(&gen_tokens(source));
            let stmts = parser.parse()?;
            let mut interpreter = Interpreter::new();
            interpreter.interpret(&stmts)
        }
    }
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

    #[test]
    fn test_evaluate_unary_expr() {
        let expr_val = do_expr("-10")
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

    #[test]
    fn test_evaluate_runtime_error() {
        let expr_val = do_expr("10 / 0")
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

    #[test]
    fn test_intepreter() {
        let mut do_interpreter = DoIt{};
        do_interpreter.interpret(
            r"print 1 + 2;
              print 4 + 6;
              print 10 + 5;
              ");
    }

    #[test]
    fn test_var_decl() {
        let mut do_interpreter = DoIt{};
        // TODO: change the Interpreter allow option output capture to endpoint instead
        // of just output to stdout. so that the unit test can capture the output
        do_interpreter.interpret(
            r"var a = 10;
              var b = a + 12;
              var c;
              print b;
              print c;
              ");
    }
}
