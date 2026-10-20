// SPDX-License-Identifier: MIT

//! PolyScribe plugin protocol (PSP/1): JSON-RPC requests over NDJSON stdio.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC request line sent from the host to a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub method: String,
    pub params: Option<Value>,
}

/// Alias used by plugins for the request type.
pub type JsonRpcRequest = Request;

/// A JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorObj {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Plugin capabilities, reported by `polyscribe-plugin-<name> info`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub name: String,
    pub version: String,
    pub protocol: String,
    pub role: String,
    pub commands: Vec<String>,
}

/// The kind of a stream item emitted by a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamItemKind {
    Progress,
    Result,
    Error,
}

/// One NDJSON line emitted by a plugin while serving a request: progress
/// updates, the final result, or an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamItem {
    #[serde(rename = "type")]
    pub kind: StreamItemKind,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObj>,
}

impl StreamItem {
    /// A progress update (percent, optional stage, optional message).
    pub fn progress(percent: f32, stage: Option<String>, message: Option<String>) -> Self {
        Self {
            kind: StreamItemKind::Progress,
            id: String::new(),
            progress: Some(percent),
            stage,
            message,
            result: None,
            error: None,
        }
    }

    /// The final successful result for a request.
    pub fn ok(id: impl Into<String>, result: Value) -> Self {
        Self {
            kind: StreamItemKind::Result,
            id: id.into(),
            progress: None,
            stage: None,
            message: None,
            result: Some(result),
            error: None,
        }
    }

    /// An error for a request.
    pub fn err(
        id: impl Into<String>,
        code: i32,
        message: impl Into<String>,
        data: Option<Value>,
    ) -> Self {
        Self {
            kind: StreamItemKind::Error,
            id: id.into(),
            progress: None,
            stage: None,
            message: None,
            result: None,
            error: Some(ErrorObj {
                code,
                message: message.into(),
                data,
            }),
        }
    }
}
