#[derive(Debug, Clone)]
pub enum Directory {
    User(String),
    Internal(String),
}

impl Default for Directory {
    fn default() -> Self {
        Self::Internal(String::new())
    }
}

impl Directory {
    #[must_use] pub fn user(path: &str) -> Self {
        Self::User(path.to_string())
    }

    #[must_use] pub fn internal(path: &str) -> Self {
        Self::Internal(path.to_string())
    }

    #[must_use] pub fn path(&self) -> String {
        match self {
            Self::User(path) => path.clone(),
            Self::Internal(path) => path.clone(),
        }
    }

    #[must_use] pub fn exists(&self) -> bool {
        match self {
            Self::User(path) => std::fs::metadata(path).is_ok(),
            Self::Internal(path) => std::fs::metadata(path).is_ok(),
        }
    }
}
