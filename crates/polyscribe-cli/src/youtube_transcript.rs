// SPDX-License-Identifier: MIT

use polyscribe_core::youtube::CaptionCue;

pub(crate) const MAX_CUES: usize = 100_000;
pub(crate) const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub(crate) const MAX_CUE_TEXT_BYTES: usize = 65_536;
pub(crate) const MAX_AGGREGATE_TEXT_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_SPEAKER_BYTES: usize = 1024;

pub(crate) fn normalize_and_validate_cues(cues: &mut [CaptionCue]) -> Result<(), &'static str> {
    if cues.is_empty() || cues.len() > MAX_CUES {
        return Err("cue count is outside the supported range");
    }

    let mut aggregate_text_bytes = 0usize;
    let mut previous_start = None;
    for cue in cues {
        cue.text = cue.text.replace("\r\n", "\n").replace('\r', "\n");
        while cue.text.ends_with('\n') {
            cue.text.pop();
        }

        if cue.start_ms > MAX_SAFE_INTEGER
            || cue.end_ms > MAX_SAFE_INTEGER
            || cue.end_ms < cue.start_ms
            || previous_start.is_some_and(|start| cue.start_ms < start)
            || cue.text.trim().is_empty()
            || cue.text.len() > MAX_CUE_TEXT_BYTES
            || cue.text.split('\n').any(|line| line.trim().is_empty())
            || cue
                .text
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
            || cue.speaker.as_ref().is_some_and(|speaker| {
                speaker.trim().is_empty()
                    || speaker.len() > MAX_SPEAKER_BYTES
                    || speaker.chars().any(char::is_control)
            })
        {
            return Err("cue text, speaker, or timestamps are invalid");
        }

        aggregate_text_bytes = aggregate_text_bytes
            .checked_add(cue.text.len())
            .ok_or("aggregate cue text is too large")?;
        if aggregate_text_bytes > MAX_AGGREGATE_TEXT_BYTES {
            return Err("aggregate cue text is too large");
        }
        previous_start = Some(cue.start_ms);
    }

    Ok(())
}

pub(crate) fn valid_language_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(text: &str) -> CaptionCue {
        CaptionCue {
            start_ms: 0,
            end_ms: 1200,
            text: text.to_string(),
            speaker: None,
            speaker_marker: false,
        }
    }

    #[test]
    fn line_endings_are_normalized_without_flattening_multiline_text() {
        let mut cues = [cue("hello\r\nworld\r\n\r\n")];
        normalize_and_validate_cues(&mut cues).unwrap();
        assert_eq!(cues[0].text, "hello\nworld");

        let mut cues = [cue("hello\r\nworld\r\n")];
        normalize_and_validate_cues(&mut cues).unwrap();
        assert_eq!(cues[0].text, "hello\nworld");

        let mut cues = [cue("hello\r\n\r\nworld")];
        assert!(normalize_and_validate_cues(&mut cues).is_err());
    }

    #[test]
    fn rejects_control_characters_blank_lines_and_unsafe_speakers() {
        for text in [
            "hello\n\nworld",
            "hello\u{1b}[31m",
            "hello\u{85}world",
            "\0",
        ] {
            let mut cues = [cue(text)];
            assert!(normalize_and_validate_cues(&mut cues).is_err(), "{text:?}");
        }

        let mut cues = [cue("hello")];
        cues[0].speaker = Some("speaker\nforged".to_string());
        assert!(normalize_and_validate_cues(&mut cues).is_err());
    }

    #[test]
    fn rejects_unsafe_timestamps_and_nonmonotonic_cues() {
        let mut cues = [cue("first"), cue("second")];
        cues[1].start_ms = 0;
        cues[0].start_ms = 1;
        assert!(normalize_and_validate_cues(&mut cues).is_err());

        let mut cues = [cue("unsafe")];
        cues[0].start_ms = MAX_SAFE_INTEGER + 1;
        assert!(normalize_and_validate_cues(&mut cues).is_err());
    }
}
