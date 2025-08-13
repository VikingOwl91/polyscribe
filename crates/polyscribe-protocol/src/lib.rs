// SPDX-License-Identifier: MIT
// PolyScribe Protocol (PSP/1): JSON-RPC 2.0 over NDJSON on stdio

use serde::{Deserialize, Serialize};

/// Plugin capabilities as reported by `--capabilities`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub name: String,
    pub version: String,
    /// Protocol identifier (e.g., "psp/1")
    pub protocol: String,
    /// Role (e.g., pipeline, tool, generator)
    pub role: String,
    /// Supported command names
    pub commands: Vec<String>,
}

/// Generic JSON-RPC 2.0 request for PSP/1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String, // "2.0"
    pub id: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// Error object for JSON-RPC 2.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Generic JSON-RPC 2.0 response for PSP/1
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum StreamItem {
    /// Progress notification (out-of-band in stream, not a JSON-RPC response)
    Progress(Progress),
    /// A proper JSON-RPC response with a result
    Result(JsonRpcResponse),
}

/// JSON-RPC 2.0 Response envelope containing either result or error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String, // "2.0"
    pub id: String,
    #[serde(flatten)]
    pub outcome: JsonRpcOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcOutcome {
    Ok { result: serde_json::Value },
    Err { error: JsonRpcError },
}

/// Progress event structure for PSP/1 streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    /// 0..=100
    pub pct: u8,
    /// Short phase name
    pub stage: Option<String>,
    /// Human-friendly detail
    pub message: Option<String>,
}

/// Convenience helpers to build items
impl StreamItem {
    pub fn progress(pct: u8, stage: impl Into<Option<String>>, message: impl Into<Option<String>>) -> Self {
        StreamItem::Progress(Progress { pct, stage: stage.into(), message: message.into() })
    }
    pub fn ok(id: impl Into<String>, result: serde_json::Value) -> Self {
        StreamItem::Result(JsonRpcResponse { jsonrpc: "2.0".into(), id: id.into(), outcome: JsonRpcOutcome::Ok { result } })
    }
    pub fn err(id: impl Into<String>, code: i64, message: impl Into<String>, data: Option<serde_json::Value>) -> Self {
        StreamItem::Result(JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: id.into(),
            outcome: JsonRpcOutcome::Err { error: JsonRpcError { code, message: message.into(), data } },
        })
    }
}
