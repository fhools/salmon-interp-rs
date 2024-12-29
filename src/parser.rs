use super::lex::{Token, TokenType};
pub(crate) struct Parser {
    tokens: Vec<Token>,

    // index into tokens vector
    current: usize
}

impl Parser {

    fn new(tokens: &Vec<Token>) -> Self {
        Parser {
            tokens: tokens.clone(),
            current: 0
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn previous(&self) ->  &Token {
        &self.tokens[self.current-1]
    }

    fn match_tok(&mut self, candidate_toks: &[TokenType]) -> bool {
        for t in candidate_toks {
            if self.check(t) {
                self.advance();
                return true
            }
        }
        false
    }

    fn check(&mut self, tok_type: &TokenType) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().token_type == *tok_type
    }

    fn is_at_end(&mut self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    fn peek(&mut self) -> &Token {
        &self.tokens[self.current]
    }
    
}

mod test {

    use super::super::lex::{Scanner, TokenType};
    use super::*;

    fn create_tokens() -> Vec<Token> {
        let mut scanner = Scanner::new("this is a test");
        scanner.scan_tokens()
    }

    fn gen_tokens<S: Into<String>>(source: S) -> Vec<Token> {
        let mut scanner = Scanner::new(source.into());
        scanner.scan_tokens()
    }

    fn num_tok(n: f32) -> Token {
        Token::new(TokenType::Number(n), n.to_string(), 1)
    }

    #[test]
    fn test_parse_cursor() {
        let mut parser = Parser::new(&create_tokens());
        let t = parser.peek();
        assert_eq!(t.token_type, TokenType::This);
        parser.advance();
        let t = parser.peek();
        assert_eq!(t.token_type, TokenType::Identifier);
    }

    #[test]
    fn test_parser_match() {
        let mut parser = Parser::new(&gen_tokens("1 2 3 4"));
        let matched = parser.match_tok(&[num_tok(1.0).token_type]);
        assert!(matched);
    }
}

