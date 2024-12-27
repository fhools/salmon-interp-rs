#![warn(clippy::all)]
use std::env;
use std::io::{self, Read} ;
use std::path::Path;
use std::fs::read_to_string;
use interp::SalmonInterp; 

use thiserror::Error;

#[derive(Error, Debug)]
enum SalmonError {
    #[error("SalmonError: error: {0}")] 
    General(String),
}

impl From<io::Error> for SalmonError {
    fn from(error: io::Error) -> Self {
        let m = format!("io error: {}", error);
        SalmonError::General(m)
    }
}

fn main() -> Result<(), SalmonError> {
    let args : Vec<String> = env::args().collect();
    match args.len() {
        1 =>  run_prompt(),
        2 => run_file(&args[1]),
        _ => {
            println!("Usage: salmon <file>");
            Err(SalmonError::General("invalid arguments".to_string()))
        }
    }
}

fn run_file<T: AsRef<Path>>(filepath: T) -> Result<(), SalmonError> {
    let data = read_to_string(filepath)?;
    println!("contents");
    println!("{}", data);
    Ok(())
}

fn run_prompt() -> Result<(), SalmonError> {
    println!("Running with REPL:");
    loop {
        let mut line : String = String::new(); 
        io::stdin().read_to_string(&mut line)?;
        if line.is_empty() {
            break;
        }
        println!("input: {}", line);
        println!("executed");
    }

    println!("Exiting...");
    Ok(())
}
