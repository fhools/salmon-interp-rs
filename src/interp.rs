use crate::expr::*;
use crate::lex::TokenType;
use std::collections::HashMap;
use std::io::{self, Write};
pub struct Interpreter<W: Write> {
    pub environment: Environment,
    out: W,
}

#[derive(Default, Debug, Clone)]
pub struct Environment {
    pub enclosing_env: Option<Box<Environment>>,
    pub values: HashMap<String, LoxValue>,
}

impl Environment {
    fn with_enclosing(enclosing_env: Environment) -> Self {
        Environment {
            enclosing_env: Some(Box::new(enclosing_env)),
            ..Environment::default()
        }
    }
    fn get(&self, source: impl AsRef<str>) -> Result<LoxValue, RuntimeError> {
        let source: &str = source.as_ref();

        // which one is better method 1 using nested chains
        // i kind of dislike the way map_or_else or map_or take the error/default value as first
        // parameter, and success value as second

        //self.values.get(source)
        //    .map_or_else(||  {
        //        self.enclosing_env.as_ref()
        //            .map_or_else(|| {
        //                Err(RuntimeError::General(Box::leak(format!("undefined variable {}", source).into_boxed_str())))
        //            },
        //            |enclosing|
        //            enclosing.get(source)
        //        )
        //    },
        //    |val| Ok(val.to_owned()))

        // or this one using early return?
        if let Some(value) = self.values.get(source) {
            return Ok(value.clone());
        }

        if let Some(ref enclosing) = self.enclosing_env {
            return enclosing.get(source);
        }
        Err(RuntimeError::General(Box::leak(
            format!("undefined variable {}", source).into_boxed_str(),
        )))
    }

    fn set(&mut self, name: impl AsRef<str>, value: LoxValue) -> Result<(), RuntimeError> {
        self.values.insert(name.as_ref().to_owned(), value);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeError {
    General(&'static str),
}
impl Interpreter<std::io::Stdout> {
    pub fn new() -> Self {
        Interpreter {
            environment: Environment::default(),
            out: std::io::stdout(),
        }
    }
}
impl<W: Write> Interpreter<W> {
    pub fn new_with_out(out_stream: W) -> Self {
        Interpreter {
            environment: Environment::default(),
            out: out_stream,
        }
    }

    fn evaluate(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        self.visit_expr(expr)
    }

    pub fn interpret(&mut self, statements: &Vec<Stmt>) -> Result<(), RuntimeError> {
        for stmt in statements {
            self.execute(stmt)?;
        }
        Ok(())
    }

    fn print_statement_output(&mut self, msg: &str) -> io::Result<()> {
        // TODO: should print to stdout and an optional output stream
        writeln!(self.out, "{}", msg)
    }

    fn execute(&mut self, stmt: &Stmt) -> Result<(), RuntimeError> {
        match stmt {
            Stmt::Expression(expr) => self.evaluate(expr).map_or_else(Err, |_v| Ok(())),
            Stmt::Print(expr) => {
                let loxval = self.evaluate(expr);
                match loxval {
                    Ok(loxval) => {
                        self.print_statement_output(loxval.to_string().as_ref())?; 
                        Ok(())
                    }
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
                //eprintln!("declaring var: {} value: {}", name, lox_value.to_string());
                self.environment.set(name, lox_value)?;
                Ok(())
            }

            Stmt::Block(ref block) => self.execute_block(
                &block.statements,
                Environment::with_enclosing(self.environment.clone()),
            ),

            Stmt::If(ref if_stmt) => {
                let lox_val = self.evaluate(&if_stmt.conditional);
                lox_val.map_or_else(
                    Err,
                    |lox_val| {
                        if is_truthy(&lox_val) {
                            self.execute(&if_stmt.then_branch)?;
                        } else if let Some(ref else_branch) = if_stmt.else_branch {
                            self.execute(else_branch)?;
                        }
                        Ok(())
                    },
                )
            }
        }

    }

    fn execute_block(&mut self, stmts: &Vec<Stmt>, env: Environment) -> Result<(), RuntimeError> {
        // save off Interpreters prior environment
        let previous = self.environment.clone();

        // set the blocks environment
        self.environment = env;
        let mut result = Ok(());
        for s in stmts {
            result = self.execute(s);
            if result.is_err() {
                break;
            }
        }
        // restore the prior environment
        self.environment = previous;
        result
    }
}

fn to_op_fn(
    op_tok: TokenType,
) -> Box<dyn Fn(&LoxValue, &LoxValue) -> Result<LoxValue, RuntimeError>> {
    Box::new(move |l, r| {
        match (&op_tok, l, r) {
            (TokenType::Plus, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval + rval))
            }
            (TokenType::Plus, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::String(format!("{}{}", lval, rval)))
            }
            (TokenType::Plus, _, _) => Err(RuntimeError::General(
                "+ operands must be both numbers or strings only",
            )),
            (TokenType::Minus, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval - rval))
            }
            (TokenType::Minus, _, _) => {
                Err(RuntimeError::General("- operands must be both numbers"))
            }
            // multiply
            (TokenType::Star, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Number(lval * rval))
            }
            (TokenType::Star, _, _) => {
                Err(RuntimeError::General("* operands must be both numbers"))
            }
            // divide
            (TokenType::Slash, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                if *rval == 0.0 {
                    Err(RuntimeError::General("divide by zero"))
                } else {
                    Ok(LoxValue::Number(lval / rval))
                }
            }
            (TokenType::Slash, _, _) => {
                Err(RuntimeError::General("/ operands must be both numbers"))
            }

            // ==
            (TokenType::EqualEqual, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval == rval))
            }

            (TokenType::EqualEqual, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval == rval))
            }

            (TokenType::EqualEqual, _, _) => Err(RuntimeError::General(
                "== operands must be both numbers, or both strings",
            )),

            // !=
            (TokenType::BangEqual, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval != rval))
            }

            (TokenType::BangEqual, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval != rval))
            }

            (TokenType::BangEqual, _, _) => Err(RuntimeError::General(
                "== operands must be both numbers, or both strings",
            )),
            _ => Err(RuntimeError::General("binary expression unknown operator")),
        }
    })
}

fn is_truthy(loxval: &LoxValue) -> bool {
    match loxval {
        LoxValue::Nil => false,
        LoxValue::Bool(ref b) => *b,
        LoxValue::Number(_) | LoxValue::String(_) => true,
    }
}

impl<W: Write> ExprVisitor<Result<LoxValue, RuntimeError>> for Interpreter<W> {
    fn visit_expr(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        match expr {
            Expr::Binary(b) => {
                let op_fn = to_op_fn(b.op.token_type.clone());
                let lvalue = self.visit_expr(&b.left);
                let rvalue = self.visit_expr(&b.right);
                //eprintln!("visit_expr binary: {:?}, {:?}", lvalue, rvalue);
                match (lvalue, rvalue) {
                    (Ok(ref lv), Ok(ref rv)) => op_fn(lv, rv),
                    _ => Err(RuntimeError::General("binary op failed on values")),
                }
            }
            Expr::Logical(logical_expr) => self.visit_logical_expr(expr),
            Expr::Literal(lit) => Ok(lit.val.clone()),
            Expr::Unary(UnaryExpr { op, ref expr }) => {
                let loxval = self.visit_expr(expr);
                //eprintln!("visit_expr unary op: {:?} inner expr: {:?}", op, loxval);
                match (&op.token_type, loxval) {
                    (TokenType::Minus, Ok(LoxValue::Number(value))) => Ok(LoxValue::Number(-value)),
                    (TokenType::Bang, Ok(ref loxval)) => Ok(LoxValue::Bool(!is_truthy(loxval))),
                    _ => Err(RuntimeError::General("unary error")),
                }
            }
            Expr::Variable(VariableExpr { ref name }) => {
                self.environment.get(&name.lexeme).map_err(|_err| {
                    eprintln!("unfound variable: {}", name);
                    RuntimeError::General(Box::leak(
                        format!("undefined variable: {}", name).into_boxed_str(),
                    ))
                })
            }
            Expr::Assign(ref assign_expr) => {
                if self.environment.get(&assign_expr.name.lexeme).is_ok() {
                    self.evaluate(&assign_expr.value)
                        .and_then(|val| {
                            eprintln!("assign var {} = {:?}", assign_expr.name.lexeme, val);
                            self.environment
                                .set(&assign_expr.name.lexeme, val.clone())?;
                            Ok(val)
                        })
                        .map_err(|_err| {
                            eprintln!("undefined variable for assignment");
                            RuntimeError::General(Box::leak(
                                format!(
                                    "undefined variable for assignment: {}",
                                    assign_expr.name.lexeme
                                )
                                .into_boxed_str(),
                            ))
                        })
                } else {
                    self.environment.get(&assign_expr.name.lexeme)
                }
            }

            a @ _ => {
                eprintln!("unhandled expr: {:?}", a);
                Err(RuntimeError::General("unhandled expr"))
            }
        }
    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        match expr {
            Expr::Logical(logical_expr) => {
                let left_val = self.evaluate(&logical_expr.left)?;
                let is_or = logical_expr.op.token_type == TokenType::Or;

                if (is_or && is_truthy(&left_val)) || (!is_or && !is_truthy(&left_val)) {
                    return Ok(left_val);
                }
                self.evaluate(&logical_expr.right)
            }
            _ => Err(RuntimeError::General(Box::leak(
                format!("visit_logical unhandled expr: {:?}", expr.to_string()).into_boxed_str(),
            ))),
        }
    }
}

fn evaluate_expr(expr: &Expr) -> Result<LoxValue, RuntimeError> {
    let mut interpreter = Interpreter::new();
    interpreter.evaluate(expr).map_or_else(
        |RuntimeError::General(s)| {
            eprintln!("runtime error to evaluate: \"{}\" : {}", expr, s);
            Err(RuntimeError::General(s))
        },
        Ok,
    )
}
mod test {
    use super::super::lex::gen_tokens;
    use super::super::parser::{do_expr, Parser};
    use super::*;

    struct DoIt {}

    impl DoIt {
        fn interpret(&mut self, source: &str) -> Result<(), RuntimeError> {
            let mut parser = Parser::new(&gen_tokens(source));
            let stmts = parser.parse()?;
            let mut interpreter = Interpreter::new();
            interpreter.interpret(&stmts)
        }

        fn interpret_capture_output(&mut self, source: &str) -> Result<String, RuntimeError> {
            let mut parser = Parser::new(&gen_tokens(source));
            let stmts = parser.parse()?;
            let mut buff = Vec::new();
            let mut interpreter = Interpreter::new_with_out(&mut buff);
            interpreter.interpret(&stmts);
            Ok(String::from_utf8_lossy(&buff).to_string())
        }
    }
    #[test]
    fn test_evaluate_expr() {
        let expr_val = do_expr("1+2+3")
            .map_err(Into::into)
            .and_then(|expr| evaluate_expr(&expr))
            .map(|lvalue| {
                let val: f32 = match lvalue {
                    LoxValue::Number(v) => v,
                    _ => f32::NAN,
                };
                val
            })
            .unwrap_or(102.0);
        assert_eq!(expr_val.to_string(), "6");
    }

    #[test]
    fn test_evaluate_unary_expr() {
        let expr_val = do_expr("-10")
            .map_err(Into::into)
            .and_then(|expr| evaluate_expr(&expr))
            .map(|lvalue| {
                let val: f32 = match lvalue {
                    LoxValue::Number(v) => v,
                    _ => f32::NAN,
                };
                val
            })
            .unwrap_or(102.0);
        assert_eq!(expr_val, -10.0);
    }

    #[test]
    fn test_evaluate_runtime_error() {
        // TODO: do_expr returns a ParseError
        let expr_val = do_expr("10 / 0")
            // convert ParseError to RutimeError
            .map_err(Into::into)
            // evaluate to get a Result<LoxValue>
            .and_then(|expr| evaluate_expr(&expr))
            // Convert RuntimeError to LoxValue with NAN
            .or(Ok::<LoxValue, RuntimeError>(LoxValue::Number(f32::NAN)))
            // convert to f32
            .map(|lvalue| {
                let val: f32 = match lvalue {
                    LoxValue::Number(v) => v,
                    _ => f32::NAN,
                };
                val
            })
            .unwrap();
        eprintln!("expr_val: {}", expr_val);
        assert!(expr_val.is_nan())
    }

    #[test]
    fn test_intepreter() {
        let mut do_interpreter = DoIt {};
        do_interpreter.interpret(
            r"print 1 + 2;
              print 4 + 6;
              print 10 + 5;
              ",
        );
    }

    #[test]
    fn test_var_decl() {
        let mut do_interpreter = DoIt {};
        // TODO: change the Interpreter allow option output capture to endpoint instead
        // of just output to stdout. so that the unit test can capture the output
        do_interpreter.interpret(
            r"var a = 10;
              var b = a + 12;
              var c;
              print b;
              c = 100 + b;
              10 + 40;
              print c;
              ",
        );
    }

    #[test]
    fn test_lexical_scope() {
        let mut do_interpreter = DoIt {};
        do_interpreter.interpret(
            r"var a = true;
              print a;
              {
                var a = 20;
                print a;
              }
              print a;
              ",
        );
    }

    #[test]
    fn test_lexical_scope_out_buff() -> Result<(), RuntimeError> {
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
              var a = 10;
              print a;
              {
                var a = 20;
                print a;
              }
              print a;
              ",
            )
            .map(|s| {
                eprintln!("captured output: {}", s);
                assert_eq!(s, "10\n20\n10\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_lexical_scope_shadow_out_buff() -> Result<(), RuntimeError> {
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
              // global a
              var a = 10;
              print a;
              {
                //shadow a but use a in initializer
                var a = a + 20;
                // expected 10 + 20 = 30
                print a;
              }
              print a;
              ",
            )
            .map(|s| {
                eprintln!("captured output: {}", s);
                assert_eq!(s, "10\n30\n10\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_logical() -> Result<(), RuntimeError> {
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
              // global a
              var a = 10;
              print a;
              if (a == 11) {
                //shadow a but use a in initializer
                var a = a + 20;
                // expected 10 + 20 = 30
                print a;
              } else {
                print 2000;
              }
              if (a == 10) {
                print 30;
              }
              ",
            )
            .map(|s| {
                eprintln!("captured output: {}", s);
                assert_eq!(s, "10\n2000\n30\n");
                Ok::<(), RuntimeError>(())
            })?
    }
}
