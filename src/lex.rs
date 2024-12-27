use crate::SalmonInterp;
use std::collections::HashMap;
use std::fmt;

use lazy_static::lazy_static;

#[derive(Debug, Clone)]
pub(crate) enum TokenType {
    // Single character
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    // One, two characters
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    // Literals
    Identifier,
    String(String),
    Number(f32),

    // Keywords
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,

    Eof,
}

lazy_static! {
    static ref KEYWORDS: HashMap<&'static str, TokenType> = {
        let mut keywords: HashMap<&'static str, TokenType> = HashMap::new();
        keywords.insert("and", TokenType::And);
        keywords.insert("class", TokenType::Class);
        keywords.insert("else", TokenType::Else);
        keywords.insert("false", TokenType::False);
        keywords.insert("for", TokenType::For);
        keywords.insert("fun", TokenType::Fun);
        keywords.insert("if", TokenType::If);
        keywords.insert("nil", TokenType::Nil);
        keywords.insert("or", TokenType::Or);
        keywords.insert("print", TokenType::Print);
        keywords.insert("return", TokenType::Return);
        keywords.insert("super", TokenType::Super);
        keywords.insert("this", TokenType::This);
        keywords.insert("true", TokenType::True);
        keywords.insert("var", TokenType::Var);
        keywords.insert("while", TokenType::While);
        keywords
    };
}
#[derive(Debug, Clone)]
pub(crate) struct Token {
    token_type: TokenType,
    lexeme: String,
    line: usize,
}

impl Token {
    pub fn new(token_type: TokenType, lexeme: String, line: usize) -> Self {
        Token {
            token_type,
            lexeme,
            line,
        }
    }

    pub fn to_string(&self) -> String {
        format!("{:?} {}", self.token_type, self.lexeme)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.to_string())
    }
}

pub(crate) struct Scanner {
    source: String,
    tokens: Vec<Token>,
    start: usize,
    current: usize,
    line: usize,
}

impl Scanner {
    fn new(source: String) -> Self {
        Scanner {
            source,
            tokens: vec![],
            start: 0,
            current: 0,
            line: 1,
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn scan_tokens(&mut self) -> Vec<Token> {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_token();
        }
        self.tokens
            .push(Token::new(TokenType::Eof, "".to_string(), self.line));
        self.tokens.clone()
    }

    fn scan_token(&mut self) {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenType::LeftParen),
            ')' => self.add_token(TokenType::RightParen),
            '{' => self.add_token(TokenType::LeftBrace),
            '}' => self.add_token(TokenType::RightBrace),
            ',' => self.add_token(TokenType::Comma),
            '.' => self.add_token(TokenType::Dot),
            '-' => self.add_token(TokenType::Minus),
            '+' => self.add_token(TokenType::Plus),
            ';' => self.add_token(TokenType::Semicolon),
            '*' => self.add_token(TokenType::Star),

            // Operators
            '!' => {
                let token_type = if self.match_char('=') {
                    TokenType::BangEqual
                } else {
                    TokenType::Bang
                };
                self.add_token(token_type);
            }
            '=' => {
                let token_type = if self.match_char('=') {
                    TokenType::EqualEqual
                } else {
                    TokenType::Equal
                };
                self.add_token(token_type);
            }
            '<' => {
                let token_type = if self.match_char('=') {
                    TokenType::LessEqual
                } else {
                    TokenType::Less
                };
                self.add_token(token_type);
            }
            '>' => {
                let token_type = if self.match_char('=') {
                    TokenType::GreaterEqual
                } else {
                    TokenType::Greater
                };
                self.add_token(token_type);
            }

            '/' => {
                if self.match_char('/') {
                    loop {
                        match self.peek() {
                            Some(c) => {
                                if c == '\n' && !self.is_at_end() {
                                    self.advance();
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    }
                } else {
                    self.add_token(TokenType::Slash);
                }
            }

            // Strings
            '"' => {
                self.string();
            }

            // Ignore whitespace
            ' ' | '\r' | '\t' => {}

            // Newline
            '\n' => {
                self.line += 1;
            }

            '0'..='9' => {
                self.number();
            }

            'a'..='z' | 'A'..='Z' | '_' => {
                self.identifier();
            }

            _ => {
                SalmonInterp::error(self.line, "unexpected char");
            }
        }
    }

    fn identifier(&mut self) {
        while let Some(ch) = self.peek() {
            if self.is_alphanumeric(ch) {
                self.advance();
            } else {
                break;
            }
        }
        let text = self.source[self.start..self.current].to_string();
        let token_type;
        if let Some(tok) = KEYWORDS.get(text.as_str()) {
            token_type = tok.to_owned();
        } else {
            token_type = TokenType::Identifier;
        }
        self.add_token(token_type)
    }
    fn number(&mut self) {
        while self.is_digit(self.peek().unwrap()) {
            self.advance();
        }

        if let Some(c) = self.peek() {
            if c == '.' {
                if let Some(c) = self.peek_next() {
                    if self.is_digit(c) {
                        self.advance();
                        while let Some(c) = self.peek() {
                            if self.is_digit(c) {
                                self.advance();
                            }
                        }
                    }
                }
            }
        }
        let number_value = self.source[self.start..self.current]
            .parse::<f32>()
            .unwrap();
        self.add_token(TokenType::Number(number_value));
    }

    fn string(&mut self) {
        while let Some(c) = self.peek() {
            println!("strings: {:?}", c);
            if c == '"' {
                break;
            }
            if c == '\n' {
                self.line += 1
            }
            if c != '"' {
                self.advance();
            }
        }

        if self.is_at_end() {
            SalmonInterp::error(self.line, "unterminated string.");
        }
        self.advance();
        let lexeme = self.source[self.start..self.current].to_string();
        let value = self.source[self.start + 1..self.current - 1].to_string();
        let string_tok = TokenType::String(value);
        self.add_token_lexeme(string_tok, lexeme);
    }

    fn advance(&mut self) -> char {
        let c = self.source.chars().nth(self.current);
        self.current += 1;
        c.unwrap()
    }

    fn add_token(&mut self, token_type: TokenType) {
        let lexeme = (&self.source[self.start..self.current]).to_string();
        self.tokens.push(Token::new(token_type, lexeme, self.line));
    }

    fn add_token_lexeme(&mut self, token_type: TokenType, lexeme: String) {
        self.tokens.push(Token::new(token_type, lexeme, self.line));
    }

    fn peek(&self) -> Option<char> {
        if self.is_at_end() {
            None
        } else {
            Some(self.source.chars().nth(self.current).unwrap())
        }
    }

    fn peek_next(&self) -> Option<char> {
        if self.current + 1 >= self.source.len() {
            None
        } else {
            Some(self.source.chars().nth(self.current + 1).unwrap())
        }
    }

    fn is_digit(&self, ch: char) -> bool {
        // we could use char::is_digit(), but lets do it truer to the original code
        let ord = ch as u32;
        ord >= ('0' as u32) && ord <= ('9' as u32)
    }

    fn is_alpha(&self, ch: char) -> bool {
        let ord = ch as u32;
        (ord >= ('a' as u32) && ord <= ('z' as u32))
            || (ord >= ('A' as u32) && ord <= ('Z' as u32))
            || (ord == '_' as u32)
    }

    fn is_alphanumeric(&self, ch: char) -> bool {
        self.is_alpha(ch) || self.is_digit(ch)
    }

    fn match_char(&mut self, ch: char) -> bool {
        if self.is_at_end() {
            return false;
        } else if self.source.chars().nth(self.current).unwrap() != ch {
            return false;
        }
        self.current += 1;
        return true;
    }
}

mod test {
    use super::Scanner;

    #[test]
    #[ignore]
    fn scanner_single_char() {
        let mut scanner = Scanner::new("+ + - ".to_string());
        let tokens = scanner.scan_tokens();
        println!("tokens: {:?}", tokens);
        assert!(tokens.len() > 0);
    }
    #[test]
    #[ignore]
    fn scanner_string() {
        let mut scanner = Scanner::new("\"test\"".to_string());
        let tokens = scanner.scan_tokens();
        println!("tokens: {:?}", tokens);
        assert!(tokens.len() > 0);
    }
    #[test]
    fn scanner_number() {
        let mut scanner = Scanner::new("10.50".to_string());
        let tokens = scanner.scan_tokens();
        println!("tokens: {:?}", tokens);
        assert!(tokens.len() > 0);
    }
    #[test]
    fn scanner_identifier() {
        let mut scanner = Scanner::new("this is a test".to_string());
        let tokens = scanner.scan_tokens();
        println!("tokens: {:?}", tokens);
        assert!(tokens.len() > 0);
    }
}
