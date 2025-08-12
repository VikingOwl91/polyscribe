// Centralized UI helpers for interactive prompts.
// Uses cliclack for consistent TTY-friendly UX.
//
// If you need a new prompt type, add it here so callers don't depend on a specific library.

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

// Test-visible counter to detect accidental prompt calls in non-interactive/CI contexts.
static PROMPT_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Reset the internal prompt call counter (testing aid).
pub fn testing_reset_prompt_call_counters() {
    PROMPT_CALLS.store(0, Ordering::Relaxed);
}

/// Get current prompt call count (testing aid).
pub fn testing_prompt_call_count() -> usize {
    PROMPT_CALLS.load(Ordering::Relaxed)
}

fn note_prompt_call() {
    PROMPT_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Prompt the user for a free-text value with a default fallback.
///
/// - Uses cliclack Input to render a TTY-friendly prompt.
/// - Returns `default` when the user submits an empty value.
/// - On any prompt error (e.g., non-TTY, read error), returns an error; callers should
///   handle it and typically fall back to `default` in non-interactive contexts.
pub fn prompt_text(prompt: &str, default: &str) -> Result<String> {
    note_prompt_call();
    let res: Result<String, _> = cliclack::input(prompt)
        .default_input(default)
        .interact();
    let value = res.map_err(|e| anyhow!("prompt error: {e}"))?;

    let trimmed = value.trim();
    Ok(if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    })
}

/// Ask for yes/no confirmation with a default choice.
///
/// Returns the selected boolean. Any underlying prompt error is returned as an error.
pub fn prompt_confirm(prompt: &str, default: bool) -> Result<bool> {
    note_prompt_call();
    let res: Result<bool, _> = cliclack::confirm(prompt)
        .initial_value(default)
        .interact();
    res.map_err(|e| anyhow!("prompt error: {e}"))
}

/// Single-select from a list of displayable items, returning the selected index.
///
/// - `items`: non-empty slice of displayable items.
/// - Returns the index into `items`.
pub fn prompt_select_index<T: std::fmt::Display>(prompt: &str, items: &[T]) -> Result<usize> {
    if items.is_empty() {
        return Err(anyhow!("prompt_select_index called with empty items"));
    }
    note_prompt_call();
    let mut sel = cliclack::select(prompt);
    for (i, it) in items.iter().enumerate() {
        sel = sel.item(i, format!("{}", it), "");
    }
    let idx: usize = sel
        .interact()
        .map_err(|e| anyhow!("prompt error: {e}"))?;
    Ok(idx)
}

/// Single-select from a list of clonable displayable items, returning the chosen item.
pub fn prompt_select_one<T: std::fmt::Display + Clone>(prompt: &str, items: &[T]) -> Result<T> {
    let idx = prompt_select_index(prompt, items)?;
    Ok(items[idx].clone())
}

/// Multi-select from a list, returning the selected indices.
///
/// - `defaults`: indices that should be pre-selected.
pub fn prompt_multiselect_indices<T: std::fmt::Display>(
    prompt: &str,
    items: &[T],
    defaults: &[usize],
) -> Result<Vec<usize>> {
    if items.is_empty() {
        return Err(anyhow!("prompt_multiselect_indices called with empty items"));
    }
    let mut ms = cliclack::multiselect(prompt);
    for (i, it) in items.iter().enumerate() {
        ms = ms.item(i, format!("{}", it), "");
    }
    note_prompt_call();
    let indices: Vec<usize> = ms
        .initial_values(defaults.to_vec())
        .required(false)
        .interact()
        .map_err(|e| anyhow!("prompt error: {e}"))?;
    Ok(indices)
}

/// Multi-select from a list, returning the chosen items in order of appearance.
pub fn prompt_multiselect<T: std::fmt::Display + Clone>(
    prompt: &str,
    items: &[T],
    defaults: &[usize],
) -> Result<Vec<T>> {
    let indices = prompt_multiselect_indices(prompt, items, defaults)?;
    Ok(indices.into_iter().map(|i| items[i].clone()).collect())
}
