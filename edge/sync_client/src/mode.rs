use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    Periodic,
    Opportunistic,
}

impl SyncMode {
    pub fn from_env() -> Self {
        match env::var("SYNC_MODE")
            .unwrap_or_else(|_| "periodic".to_string())
            .to_lowercase()
            .as_str()
        {
            "opportunistic" => SyncMode::Opportunistic,
            _ => SyncMode::Periodic,
        }
    }
}
