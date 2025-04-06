
#[derive(Debug, Clone)]
pub enum Directory{
    User(String),
    Internal(String),
}

impl Default for Directory {
    fn default() -> Self {
        Directory::Internal("".to_string())
    }
}

impl Directory {
    pub fn user(path: &str) -> Self {
        Directory::User(path.to_string())
    }

    pub fn internal(path: &str) -> Self {
        Directory::Internal(path.to_string())
    }

    pub fn path(&self) -> String {
        match self {
            Directory::User(path) => path.clone(),
            Directory::Internal(path) => path.clone(),
        }
    }

    pub fn exists(&self) -> bool {
        match self {
            Directory::User(path) => std::fs::metadata(path).is_ok(),
            Directory::Internal(path) => std::fs::metadata(path).is_ok(),
        }
    }
}