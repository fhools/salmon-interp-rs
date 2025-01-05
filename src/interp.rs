use crate::expr::*;
use crate::lex::{Token, TokenType};
use crate::resolver::Resolver;
use std::collections::HashMap;
use std::io::{self, Write, Cursor};
use std::any::Any;

// this AnyWrite/WriteAny is to allow us to coerce the Write dyn object into a Cursor
pub trait WriteAny: Write + Any {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
type AnyWrite = dyn WriteAny;

// Blanket impl for T: Write + Any
impl<T: Write + Any> WriteAny for T {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
pub struct Interpreter {

    // there is always 1 environment allocated at the start with id 0 which is the global env
    pub environments: Vec<Environment>,
    pub cur_env: usize,
    

    // Locals var distance map for variable scope resolution
    // key: expr.id var: scope distance 
    locals: HashMap<usize, usize>,
    out: Box<AnyWrite>,
}

#[derive(Default, Debug)]
pub struct Environment {
    // enclosing_env is an index id  into the Interpreter::environments vector
    pub enclosing_env: Option<usize>,
    pub values: HashMap<String, LoxValue>,
}

impl Environment {
    fn with_enclosing(enclosing_env_id: usize) -> Self {
        Environment {
            enclosing_env: Some(enclosing_env_id),
            ..Environment::default()
        }
    }

}

#[derive(Debug, Clone)]
pub enum RuntimeError {
    General(&'static str),
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            environments: vec![Environment::default()],
            cur_env: 0,
            locals: HashMap::new(),
            out: Box::new(std::io::stdout()),
        }
    }

    pub fn new_with_out(out_stream: Box<AnyWrite>) -> Self {
        Interpreter {
            environments: vec![Environment::default()],
            cur_env: 0,
            locals: HashMap::new(),
            out: out_stream,
        }
    }

    pub fn global_env_id(&self) -> usize {
        0
    }
    fn evaluate(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        self.visit_expr(expr)
    }

    pub fn resolve(&mut self, expr: &Expr, depth: usize) {
        
        self.locals.insert(expr.id, depth);
    }

    fn lookup_variable(&mut self, name: &Token, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        let distance = self.locals.get(&expr.id);

        // The scoping distance to the variable is found by the resolver after parsing, prior to
        // execution. During runtime it should find the local variable, otherwise its a global
        if let Some(distance) = distance {
            //eprintln!("distance of {} is {} of expr: {}", name.lexeme, distance, expr.to_string());
            self.get_at(*distance, &name.lexeme)
        } else {
            self.get(&name.lexeme, self.global_env_id(), 0)
        }
    }
    pub fn interpret(&mut self, statements: &Vec<Stmt>) -> Result<LoxValue, RuntimeError> {
        let mut result = Err(RuntimeError::General("intpret fail")); 
        for stmt in statements {
            result = self.execute(stmt);
        }
        result
    }

    fn print_statement_output(&mut self, msg: &str) -> io::Result<()> {
        // TODO: should print to stdout and an optional output stream
        writeln!(self.out, "{}", msg)
    }

    fn execute(&mut self, stmt: &Stmt) -> Result<LoxValue, RuntimeError> {
        match stmt {
            Stmt::Expression(expr) => self.evaluate(expr).map_or_else(Err, |v| Ok(v)),
            Stmt::Print(expr) => {
                let loxval = self.evaluate(expr);
                match loxval {
                    Ok(loxval) => {
                        self.print_statement_output(loxval.to_string().as_ref())?; 
                        Ok(loxval)
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
                self.define(name, self.cur_env, lox_value.clone())?;
                Ok(lox_value)
            }

            Stmt::Block(ref block) => {
                //eprintln!("block running");
                self.execute_block(&block.statements, self.cur_env)
            },

            Stmt::If(ref if_stmt) => {
                let lox_val = self.evaluate(&if_stmt.conditional)?;
                if is_truthy(&lox_val) {
                    return self.execute(&if_stmt.then_branch)
                } else if let Some(ref else_branch) = if_stmt.else_branch {
                    return self.execute(else_branch)
                } else {
                    return Ok(LoxValue::Nil);
                }
            },

            Stmt::While(ref while_stmt) => {
                let mut cond_lox_val = self.evaluate(&while_stmt.condition)?;
                while is_truthy(&cond_lox_val) {
                    let stmt_val = self.execute(&while_stmt.body)?;
                    if let LoxValue::Return(_) = stmt_val {
                        return Ok(stmt_val);
                    }
                    cond_lox_val = self.evaluate(&while_stmt.condition)?;
                }
                Ok(LoxValue::Nil)
            },
            Stmt::Function(ref function_stmt) => { 
                let closure_id = self.push_env(self.cur_env);
                let new_function = LoxFunction{ 
                    function: Box::new(function_stmt.clone()), 
                    closure: closure_id};
                let loxval = LoxValue::Function(Box::new(new_function));
                self.define(&function_stmt.name.lexeme, self.cur_env, loxval);
                Ok(LoxValue::Nil)
            },
            Stmt::Return(ref return_stmt)  => {
                match return_stmt.value {
                    Some(ref expr) => { 
                        let return_val = self.evaluate(expr)?;
                        Ok(LoxValue::Return(Some(Box::new(return_val))))
                    },
                    None => {
                        Ok(LoxValue::Return(None))
                    }
                }
            }
            Stmt::ParseError => { todo!() }
        }

    }

    pub fn execute_block(&mut self, stmts: &Vec<Stmt>, enclosing_env_id: usize) -> Result<LoxValue, RuntimeError> {
        // save off Interpreters prior environment
        let previous = self.cur_env;
        
        let new_env = self.push_env(enclosing_env_id);
        self.cur_env = new_env;

        // TODO: what should the default value be for execute_block? currently it maybe the last
        // value of last statement executed
        let mut result = Ok(LoxValue::Nil);
        for s in stmts {
            //eprintln!("going to run: {:?}", s);
            result = self.execute(s);
            if result.is_err() {
                eprintln!("error running a block statement: {:?}", result);
                break;
            } else if let Ok(LoxValue::Return(_)) = result {
                // need to break instead of returning so that the env is restored
                break;
            }
        }
        // restore the prior environment
        self.cur_env = previous;
        self.pop_env();
        result
    }

    fn ancestor_env_id(&mut self, distance: usize, mut this_env: usize) -> usize {
        let mut env: &mut Environment = self.environments.get_mut(this_env).unwrap();
        let mut i = 0;
        while i < distance {
            if let Some(enclosing_env_id) = env.enclosing_env {
                this_env = enclosing_env_id;
            }
            i += 1;
        }
        this_env
    }

    fn get_at(&mut self, distance: usize, source: impl AsRef<str>) -> Result<LoxValue, RuntimeError> {
        let ancestor_env_id = self.ancestor_env_id(distance, self.cur_env);
        self.get(source, ancestor_env_id, 0)
    }

    // DEBUG add level parameter
    fn get(&self, source: impl AsRef<str>, env_id: usize, level: usize) -> Result<LoxValue, RuntimeError> {
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
        //eprintln!("looking in env_id {}", env_id);
        if let Some(value) = self.environments[env_id].values.get(source) {
            //eprintln!("env get level: {} key: {}  val: {}", level, source, value.to_string());
            return Ok(value.clone());
        }
        if level == 20 {
            panic!("too many levels");
        }
        if let Some(ref enclosing_id) = self.environments[env_id].enclosing_env {
            //eprintln!("did not find in env_id {} looking in id {}", env_id, enclosing_id);
            return self.get(source, *enclosing_id,level + 1);
        }
        Err(RuntimeError::General(Box::leak(
            format!("undefined variable {}", source).into_boxed_str(),
        )))
    }

    pub fn define(&mut self, name: impl AsRef<str>, env_id: usize, value: LoxValue) -> Result<(), RuntimeError> {
        self.environments[env_id].values.insert(name.as_ref().to_owned(), value);
        Ok(())
    }

    fn assign(&mut self, name: impl AsRef<str>, env_id: usize,  value: LoxValue, level: usize) -> Result<(), RuntimeError> {
        let name = name.as_ref();
        if self.environments[env_id].values.contains_key(name) {
            //eprintln!("env assign level: {} key: {} val {}", level, name, value.to_string());
            self.environments[env_id].values.insert(name.to_owned(), value);
            Ok(())
        } else if let Some(ref env_id) = self.environments[env_id].enclosing_env {
            //eprintln!("not found in env.assign level: {} key: {} val {}", level, name, value.to_string());
            self.assign(name, *env_id, value, level + 1)
        } else {
            Err(RuntimeError::General(Box::leak(
                        format!("undefined variable {}", name).into_boxed_str(),
                        )))
        }
    }

    pub fn push_env(&mut self, enclosing_env_id: usize) ->  usize {
        // NOTE: this will probably need to be addressed for closures
        self.environments.push(Environment::with_enclosing(enclosing_env_id));
        self.environments.len()-1
    }

    pub fn pop_env(&mut self) {
        // FIXME: we are implementing closures, so we cannot actually destroy any Environments
        // so this pop_env does a no-op
        //self.environments.pop();
    }
    pub fn get_buffer_contents(&mut self) -> Option<String> {
        // this is courtesy of chatgpt for this tricky part
        // to coerce the output to an any for the downcast_mut call
        let write_any_ref = self.out.as_mut();
        let any_cursor = write_any_ref.as_any_mut(); 
        if let Some(cursor) = any_cursor.downcast_mut::<std::io::Cursor<Vec<u8>>>() {
            let bytes = cursor.get_ref();
            return Some(String::from_utf8_lossy(bytes).to_string());
        }
        eprintln!("get_buffer failed");
        None
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
                "!= operands must be both numbers, or both strings",
            )),

            // <
            (TokenType::Less, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval < rval))
            }
            (TokenType::Less, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval < rval))
            }
            (TokenType::Less, _, _) => Err(RuntimeError::General(
                "< operands must be both numbers, or both strings",
            )),

            // <=
            (TokenType::LessEqual, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval <= rval))
            }
            (TokenType::LessEqual, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval <= rval))
            }
            (TokenType::LessEqual, _, _) => Err(RuntimeError::General(
                "<= operands must be both numbers, or both strings",
            )),

            // >
            (TokenType::Greater, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval > rval))
            }
            (TokenType::Greater, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval > rval))
            }
            (TokenType::Greater, _, _) => Err(RuntimeError::General(
                "> operands must be both numbers, or both strings",
            )),

            // >=
            (TokenType::GreaterEqual, LoxValue::Number(ref lval), LoxValue::Number(ref rval)) => {
                Ok(LoxValue::Bool(lval >= rval))
            }
            (TokenType::GreaterEqual, LoxValue::String(ref lval), LoxValue::String(ref rval)) => {
                Ok(LoxValue::Bool(lval >= rval))
            }
            (TokenType::GreaterEqual, _, _) => Err(RuntimeError::General(
                "> operands must be both numbers, or both strings",
            )),

            _ => Err(RuntimeError::General("binary expression unknown operator")),
        }
    })
}

fn is_truthy(loxval: &LoxValue) -> bool {
    match loxval {
        // NOTE: truthy for return should actually be a runtime error, but we'll just set it to
        // false for now
        LoxValue::Nil | LoxValue::Return(_) => false,
        LoxValue::Bool(ref b) => *b,
        LoxValue::Number(_) | LoxValue::String(_) | LoxValue::Function(_) => true,
    }
}

impl ExprVisitor<Result<LoxValue, RuntimeError>> for Interpreter {

    fn visit_expr(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        match &expr.kind {
            ExprKind::Binary(b) => {
                let op_fn = to_op_fn(b.op.token_type.clone());
                let lvalue = self.visit_expr(&b.left);
                let rvalue = self.visit_expr(&b.right);
                //eprintln!("visit_expr binary: {:?}, {:?}", lvalue, rvalue);
                match (lvalue, rvalue) {
                    (Ok(ref lv), Ok(ref rv)) => op_fn(lv, rv),
                    _ => Err(RuntimeError::General("binary op failed on values")),
                }
            }
            ExprKind::Logical(logical_expr) => self.visit_logical_expr(expr),
            ExprKind::Literal(lit) => Ok(lit.val.clone()),
            ExprKind::Unary(UnaryExpr { op, ref expr }) => {
                let loxval = self.visit_expr(expr);
                //eprintln!("visit_expr unary op: {:?} inner expr: {:?}", op, loxval);
                match (&op.token_type, loxval) {
                    (TokenType::Minus, Ok(LoxValue::Number(value))) => Ok(LoxValue::Number(-value)),
                    (TokenType::Bang, Ok(ref loxval)) => Ok(LoxValue::Bool(!is_truthy(loxval))),
                    _ => Err(RuntimeError::General("unary error")),
                }
            }
            ExprKind::Variable(VariableExpr { ref name }) => {
                //self.get(&name.lexeme,self.cur_env, 0).map_err(|_err| {
                //    eprintln!("unfound variable: {}", name);
                //    RuntimeError::General(Box::leak(
                //        format!("undefined variable: {}", name).into_boxed_str(),
                //    ))
                //})
                self.lookup_variable(&name, expr)
            }
            ExprKind::Assign(ref assign_expr) => {
                let loxval = self.evaluate(&assign_expr.value)?;
                self.assign(&assign_expr.name.lexeme, self.cur_env, loxval.clone(), 0)?;
                Ok(loxval)
                //if self.environment.get(&assign_expr.name.lexeme, 0).is_ok() {
                //    self.evaluate(&assign_expr.value)
                //        .and_then(|val| {
                //            eprintln!("assign var {} = {:?}", assign_expr.name.lexeme, val);
                //            self.environment
                //                .assign(&assign_expr.name.lexeme, val.clone(), 0)?;
                //            Ok(val)
                //        })
                //        .map_err(|_err| {
                //            eprintln!("undefined variable for assignment");
                //            RuntimeError::General(Box::leak(
                //                format!(
                //                    "undefined variable for assignment: {}",
                //                    assign_expr.name.lexeme
                //                )
                //                .into_boxed_str(),
                //            ))
                //        })
                //} else {
                //    self.environment.get(&assign_expr.name.lexeme)
                //}
            },
            ExprKind::Call(ref call_expr) => {
                let callee = self.evaluate(&call_expr.callee)?;
                let mut arguments = Vec::new();
                for expr in &call_expr.arguments {
                    let arg = self.evaluate(expr)?;
                    arguments.push(arg.clone());
                }
                // TODO: check arity of arguments matches 
                // with the callee parameters arity
                
                match callee {
                    LoxValue::Function(ref callable) => {
                        callable.call(self, &arguments)
                    },
                    _ => {
                        Err(RuntimeError::General("callee expression is not callable"))
                    }
                }

            },

            a @ _ => {
                eprintln!("unhandled expr: {:?}", a);
                Err(RuntimeError::General("unhandled expr"))
            }
        }
    }

    fn visit_logical_expr(&mut self, expr: &Expr) -> Result<LoxValue, RuntimeError> {
        match &expr.kind {
            ExprKind::Logical(logical_expr) => {
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
        fn interpret(&mut self, source: &str) -> Result<LoxValue, RuntimeError> {
            let mut parser = Parser::new(&gen_tokens(source));
            let stmts = parser.parse()?;
            let mut interpreter = Interpreter::new();
            let mut resolver = Resolver::new();
            resolver.resolve(&mut interpreter, &stmts);
            interpreter.interpret(&stmts)
        }

        fn interpret_capture_output(&mut self, source: &str) -> Result<String, RuntimeError> {
            let mut parser = Parser::new(&gen_tokens(source));
            let stmts = parser.parse()?;
            let mut cursor_buff = Cursor::new(Vec::new());
            let mut interpreter = Interpreter::new_with_out(Box::new(cursor_buff));

            let mut resolver = Resolver::new();
            resolver.resolve(&mut interpreter, &stmts);
            interpreter.interpret(&stmts);
            let output = interpreter.get_buffer_contents();
            output.map_or_else(|| Err(RuntimeError::General("error")), |s| Ok(s))
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
                // var a = a + 20 not allowed anymore because of resolver
                var b = a + 20;
                // expected 10 + 20 = 30
                print b;
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
                // shadow not allowed anymore
                // var a = a + 20;
                var b = a + 20;
                // expected 10 + 20 = 30
                print b;
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

    #[test]
    fn test_while() -> Result<(), RuntimeError> {
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
              var a = 1;
              while (a != 5) {
                 print a;
                 a = a + 1;
              }
              print a;
              ",
            )
            .map(|s| {
                eprintln!("captured output: {}", s);
                assert_eq!(s, "1\n2\n3\n4\n5\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_for() -> Result<(), RuntimeError> {
        // test for loop, also test that local for var is shadowed and then destroyed after loop
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
                var a = 10;
                for (var a = 1; a != 5;) {
                  print a;
                  a = a + 1;
                }
                print a;
              ",
            )
            .map(|s| {
                assert_eq!(s, "1\n2\n3\n4\n10\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_basic_function() -> Result<(), RuntimeError> {
        // test for loop, also test that local for var is shadowed and then destroyed after loop
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
                fun foo() {
                  for(var a = 1; a < 20; a = a + 1) {
                    if (a == 10) {
                        return a;
                    }
                  }
                  return a;
                }
                print foo();
              ",
            )
            .map(|s| {
                assert_eq!(s, "10\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_functions_fib() -> Result<(), RuntimeError> {
        // test for loop, also test that local for var is shadowed and then destroyed after loop
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
                fun fib(n) {
                    if (n <= 1) return n;
                    return fib(n-2) + fib(n-1);
                }
                print fib(7);
              ",
            )
            .map(|s| {
                assert_eq!(s, "13\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_closures() -> Result<(), RuntimeError> {
        // test for loop, also test that local for var is shadowed and then destroyed after loop
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r"
                fun makeCounter() {
                  var i = 0;
                  fun count() {
                    i = i + 1;
                    print i;
                  }
                  return count;
                }
                var counter = makeCounter();
                counter();
                counter();
              ",
            )
            .map(|s| {
                assert_eq!(s, "1\n2\n");
                Ok::<(), RuntimeError>(())
            })?
    }

    #[test]
    fn test_resolver() -> Result<(), RuntimeError> {
        // test for loop, also test that local for var is shadowed and then destroyed after loop
        let mut do_interpreter = DoIt {};
        do_interpreter
            .interpret_capture_output(
                r#"
                var a = "global";
                {
                    fun showA() {
                        print a;
                    }
                    showA();
                    var a = "block";
                    showA();
                }
              "#,
            )
            .map(|s| {
                assert_eq!(s, "global\nglobal\n");
                Ok::<(), RuntimeError>(())
            })?
    }
}
