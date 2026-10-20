#[derive(Clone, Debug)]
pub enum OutputMode {
    Json,
    Human { quiet: bool },
}

impl OutputMode {
    pub fn print_json<T: serde::Serialize>(&self, v: &T) {
        if let OutputMode::Json = self {
            // Write compact JSON to stdout without prefixes
            // and ensure a trailing newline for CLI ergonomics
            let s = serde_json::to_string(v).unwrap_or_else(|e| format!("\"JSON_ERROR:{}\"", e));
            println!("{}", s);
        }
    }
}
