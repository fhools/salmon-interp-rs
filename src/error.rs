use thiserror::Error;
use crate::lex::{Token};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::io::{self} ;

pub static ERROR_MGR: LazyLock<Mutex<Option<ErrorManager>>> = LazyLock::new(|| { Mutex::new(None) });

pub struct ErrorManager {
    
}
#[derive(Error, Debug)]
pub enum SalmonError {
    #[error("SalmonError: error: {0}")] 
    General(String),
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("GeneralError: token: {0} msg: {1}")]
    General(Token, String)
}

impl From<io::Error> for SalmonError {
    fn from(error: io::Error) -> Self {
        let m = format!("io error: {}", error);
        SalmonError::General(m)
    }
}

