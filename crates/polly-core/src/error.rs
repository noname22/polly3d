use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unknown {kind} '{id}'")]
    NotFound { kind: &'static str, id: String },
    #[error("invalid id '{0}' (expected forms like v12, f3, e3-9, o1)")]
    BadId(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(msg.into()))
}
