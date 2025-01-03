# Description
This is a implementation of the Lox interpreter from Bob Nystrom's wonderful book Crafting Interpreters

# Status
Not completed yet. Last completed chapter is Chapter 10.

Still needs to handle variable resolution for proper lexical scoping and classes.

# Rustisms
The Expr/Stmt class hierarchy is replaced by enum's. We had to use Box<Expr> and Box<Stmt> to get around
the recursive nature of the AST composition.


The Environment struct is just a plan old data type without methods, this is  because
we would have had to use Rc<RefCell<Environment> to form the linked list of Environment lookup chains.
Instead the Interpreter class holds a Vec<Environment> for all Environment instances  and the Interpreter 
contains the define() and assign() methods to work with Environment variable lookup. The enclosing 
environment pointer/reference is just a usize id that is used to index into the Vec<Environment> 

The book uses Java's root Object to store all of Lox's values and a bunch of other stuff. We instead created a
LoxValue enum that stores Number, Strings, Bool, Nil, and Return which is described below.

The Lox interpreter in the book uses Java's try/catch to implement error handling and unwind the stack
for function 'return' statements, instead this is handled by all of the Interpreter's execute methods returning a Result<LoxValue>, and where it is necessary the execution checks for LoxValue::Return variants to return from the function.

