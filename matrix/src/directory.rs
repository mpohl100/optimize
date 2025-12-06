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
    #[must_use]
    pub fn user(path: &str) -> Self {
        Self::User(path.to_string())
    }

    #[must_use]
    pub fn internal(path: &str) -> Self {
        Self::Internal(path.to_string())
    }

    #[must_use]
    pub fn path(&self) -> String {
        match self {
            Self::Internal(path) | Self::User(path) => path.clone(),
        }
    }

    #[must_use]
    pub fn path_with_workspace(
        &self,
        workspace: &str,
    ) -> String {
        match self {
            Self::Internal(path) => {
                if workspace.is_empty() {
                    path.clone()
                } else {
                    format!("{workspace}/{path}")
                }
            },
            Self::User(path) => path.clone(),
        }
    }

    #[must_use]
    pub fn exists(&self) -> bool {
        match self {
            Self::Internal(path) | Self::User(path) => std::fs::metadata(path).is_ok(),
        }
    }
}
