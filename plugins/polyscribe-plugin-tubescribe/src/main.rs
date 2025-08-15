// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use clap::Parser;
use polyscribe_protocol as psp;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};

#[derive(Parser, Debug)]
#[command(name = "polyscribe-plugin-tubescribe", version, about = "Stub tubescribe plugin for PolyScribe PSP/1")]
struct Args {
    /// Print capabilities JSON and exit
    #[arg(long)]
    capabilities: bool,
    /// Serve mode: read one JSON-RPC request from stdin, stream progress and final result
    #[arg(long)]
    serve: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.capabilities {
        let caps = psp::Capabilities {
            name: "tubescribe".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol: "psp/1".to_string(),
            role: "pipeline".to_string(),
            commands: vec!["generate_metadata".to_string()],
        };
        let s = serde_json::to_string(&caps)?;
        println!("{}", s);
        return Ok(());
    }
    if args.serve {
        serve_once()?;
        return Ok(());
    }
    let caps = psp::Capabilities {
        name: "tubescribe".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        protocol: "psp/1".to_string(),
        role: "pipeline".to_string(),
        commands: vec!["generate_metadata".to_string()],
    };
    println!("{}", serde_json::to_string(&caps)?);
    Ok(())
}

fn serve_once() -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();
    reader.read_line(&mut line).context("failed to read request line")?;
    let req: psp::JsonRpcRequest = serde_json::from_str(line.trim()).context("invalid JSON-RPC request")?;

    emit(&psp::StreamItem::progress(5, Some("start".into()), Some("initializing".into())))?;
    std::thread::sleep(std::time::Duration::from_millis(50));
    emit(&psp::StreamItem::progress(25, Some("probe".into()), Some("probing sources".into())))?;
    std::thread::sleep(std::time::Duration::from_millis(50));
    emit(&psp::StreamItem::progress(60, Some("analyze".into()), Some("analyzing".into())))?;
    std::thread::sleep(std::time::Duration::from_millis(50));
    emit(&psp::StreamItem::progress(90, Some("finalize".into()), Some("finalizing".into())))?;

    let result = match req.method.as_str() {
        "generate_metadata" => {
            let title = "Canned title";
            let description = "Canned description for demonstration";
            let tags = vec!["demo", "tubescribe", "polyscribe"];
            json!({
                "title": title,
                "description": description,
                "tags": tags,
            })
        }
        other => {
            let err = psp::StreamItem::err(req.id.clone(), -32601, format!("Method not found: {}", other), None);
            emit(&err)?;
            return Ok(());
        }
    };

    emit(&psp::StreamItem::ok(req.id.clone(), result))?;
    Ok(())
}

fn emit(item: &psp::StreamItem) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    let s = serde_json::to_string(item)?;
    stdout.write_all(s.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}
