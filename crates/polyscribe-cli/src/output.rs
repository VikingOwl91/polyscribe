use std::io::{self, Write};

#[derive(Clone, Debug)]
pub enum OutputMode {
    Json,
    Human { quiet: bool },
}

impl OutputMode {
    pub fn is_quiet(&self) -> bool {
        matches!(self, OutputMode::Json) || matches!(self, OutputMode::Human { quiet: true })
    }

    pub fn print_json<T: serde::Serialize>(&self, v: &T) {
        if let OutputMode::Json = self {
            // Write compact JSON to stdout without prefixes
            // and ensure a trailing newline for CLI ergonomics
            let s = serde_json::to_string(v).unwrap_or_else(|e| format!("\"JSON_ERROR:{}\"", e));
            println!("{}", s);
        }
    }

    pub fn print_line(&self, s: impl AsRef<str>) {
        match self {
            OutputMode::Json => {
                // Suppress human lines in JSON mode
            }
            OutputMode::Human { quiet } => {
                if !quiet {
                    let _ = writeln!(io::stdout(), "{}", s.as_ref());
                }
            }
        }
    }
}

