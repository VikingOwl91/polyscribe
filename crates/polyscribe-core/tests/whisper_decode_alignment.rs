// SPDX-License-Identifier: MIT

//! How the decoding of a window depends on its position.
//!
//! whisper does not always report a segment for the beginning of the audio
//! it is given. For some positions it reports its first segment many
//! seconds into the window, and for others it reports nothing at all. The
//! difference is a fraction of a second, and it does not depend on the
//! number of threads or on what was decoded before.
//!
//! This is why the pipeline decodes a region that whisper did not
//! transcribe again from a slightly different position: the position
//! decides whether the speech is reported.
//!
//! The test needs a Whisper model and an audio file, so it skips itself
//! without them.

use std::path::PathBuf;

fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

fn read_wav(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let mut pos = 12;
    let mut bits = 0u16;
    let (start, len) = loop {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            bits = u16::from_le_bytes([bytes[pos + 8 + 14], bytes[pos + 8 + 15]]);
        }
        if id == b"data" {
            break (pos + 8, size);
        }
        pos += 8 + size + (size % 2);
    };
    match bits {
        16 => bytes[start..start + len]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        _ => panic!("only 16 bit PCM is supported"),
    }
}

/// Decode a window and report where the first segment starts.
fn first_segment(
    state: &mut whisper_rs::WhisperState,
    params: &whisper_rs::FullParams,
    samples: &[f32],
) -> Option<(f64, usize)> {
    state.full(params.clone(), samples).ok()?;
    let mut count = 0;
    let mut first = None;
    for segment in state.as_iter() {
        if count == 0 {
            first = Some(segment.start_timestamp() as f64 * 0.01);
        }
        count += 1;
    }
    Some((first?, count))
}

#[test]
fn the_decoding_of_a_window_is_the_same_when_it_is_repeated() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the decode test");
        return;
    };
    let Ok(audio) = std::env::var("POLYSCRIBE_DECODE_TEST_AUDIO") else {
        eprintln!("SKIPPED: POLYSCRIBE_DECODE_TEST_AUDIO is not set; skipping the decode test");
        return;
    };
    let samples = read_wav(&PathBuf::from(audio));
    if samples.len() < 480_000 {
        eprintln!("SKIPPED: the audio is too short");
        return;
    }

    let context = whisper_rs::WhisperContext::new_with_params(
        &model,
        whisper_rs::WhisperContextParameters::default(),
    )
    .unwrap();
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(4);
    params.set_translate(false);
    params.set_language(Some("en"));

    // The same window, decoded again and again, gives the same result.
    // What changes it is the position of the window, not the history.
    let window = &samples[0..480_000];
    let mut state = context.create_state().unwrap();
    let first = first_segment(&mut state, &params, window).expect("the window is decoded");
    let second = first_segment(&mut state, &params, window).expect("the window is decoded");
    assert_eq!(first, second);

    // A window that starts a fraction of a second later can report a
    // segment for its beginning where the first one did not.
    let mut later = context.create_state().unwrap();
    let shifted = first_segment(&mut later, &params, &samples[4_000..484_000])
        .expect("the window is decoded");
    eprintln!("the window at 0 s: {first:?}, a quarter of a second later: {shifted:?}");
    assert!(shifted.1 > 0, "the window is transcribed");
}
