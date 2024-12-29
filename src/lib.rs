#[macro_use]
extern crate lazy_static;
use lazy_static::lazy_static;
use std::sync::Mutex;

lazy_static! {
    static ref SALMON_INTERP: Mutex<SalmonInterp> = {
        let s = SalmonInterp::new();
        Mutex::new(s)
    };
}

pub struct SalmonInterp {
    had_error: bool,
}

impl SalmonInterp {
    fn new() -> Self {
        SalmonInterp { had_error: false }
    }

    pub fn error(line: usize, message: &str) {
        SalmonInterp::report(line, "", message);
    }

    pub fn report(line: usize, w: &str, m: &str) {
        let e = format!("[line {}] Error {}: {}", line, w, m);
        println!("{}", e);
        SALMON_INTERP.lock().unwrap().had_error = true;
    }
}

mod expr;
mod lex;
mod parser;
