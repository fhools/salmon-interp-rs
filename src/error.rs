use thiserror::Error;
use crate::lex::{Token};
use crate::interp::RuntimeError;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::io::{self} ;

pub static ERROR_MGR: LazyLock<Mutex<Option<ErrorManager>>> = LazyLock::new(|| { Mutex::new(None) });

pub struct ErrorManager {
    
}
#[derive(Error, Debug, Clone)]
pub enum SalmonError {
    #[error("SalmonError: error: {0}")] 
    General(String),
}

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("GeneralError: token: {0} msg: {1}")]
    General(Token, String)
}

impl From<ParseError> for RuntimeError {
    // TODO: Store more info in ParseError to convert to RuntimeError
    fn from(ParseError::General(_, string): ParseError) -> Self {
       RuntimeError::General(Box::leak(string.into_boxed_str()) )
    }
}

impl From<io::Error> for RuntimeError {
    fn from(err: io::Error) -> Self {
       RuntimeError::General(Box::leak(err.to_string().into_boxed_str()) )
    }
}

impl From<io::Error> for SalmonError {
    fn from(error: io::Error) -> Self {
        let m = format!("io error: {}", error);
        SalmonError::General(m)
    }
}

