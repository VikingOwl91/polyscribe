use polyscribe::progress::{ProgressFactory, ProgressMode, SelectionInput, select_mode, ProgressManager};

#[test]
fn test_factory_decide_mode_none_when_disabled() {
    let pf = ProgressFactory::new(true); // force disabled
    assert!(matches!(pf.decide_mode(0), ProgressMode::None));
    assert!(matches!(pf.decide_mode(1), ProgressMode::None));
    assert!(matches!(pf.decide_mode(2), ProgressMode::None));
}

#[test]
fn test_select_mode_zero_inputs_is_none() {
    let si = SelectionInput {
        inputs_len: 0,
        no_progress_flag: false,
        stderr_tty_override: Some(true),
        env_no_progress: false,
    };
    let (enabled, mode) = select_mode(si);
    assert!(enabled);
    assert!(matches!(mode, ProgressMode::None));
}

#[test]
fn test_select_mode_one_input_is_single() {
    let si = SelectionInput {
        inputs_len: 1,
        no_progress_flag: false,
        stderr_tty_override: Some(true),
        env_no_progress: false,
    };
    let (enabled, mode) = select_mode(si);
    assert!(enabled);
    assert!(matches!(mode, ProgressMode::Single));
}

#[test]
fn test_select_mode_multi_inputs_is_multi() {
    let si = SelectionInput {
        inputs_len: 3,
        no_progress_flag: false,
        stderr_tty_override: Some(true),
        env_no_progress: false,
    };
    let (enabled, mode) = select_mode(si);
    assert!(enabled);
    match mode {
        ProgressMode::Multi { total_inputs } => assert_eq!(total_inputs, 3),
        _ => panic!("expected multi mode"),
    }
}

#[test]
fn test_env_no_progress_disables() {
    // Simulate env flag influence by passing env_no_progress=true
    unsafe { std::env::set_var("NO_PROGRESS", "1"); }
    let si = SelectionInput {
        inputs_len: 5,
        no_progress_flag: false,
        stderr_tty_override: Some(true),
        env_no_progress: true,
    };
    let (enabled, mode) = select_mode(si);
    assert!(!enabled);
    assert!(matches!(mode, ProgressMode::None));
    unsafe { std::env::remove_var("NO_PROGRESS"); }
}

#[test]
fn test_completed_never_exceeds_total_and_item_updates_do_not_affect_total() {
    // create hidden multiprogress for tests
    let pm = ProgressManager::new_for_tests_multi_hidden(3);
    pm.set_total(3);
    // Start an item and update progress a few times
    let item = pm.start_item("Test item");
    item.set_progress(0.1);
    item.set_progress(0.4);
    item.set_message("stage1");
    // Ensure total unchanged
    let (pos, len) = pm.total_state_for_tests().unwrap();
    assert_eq!(len, 3);
    assert_eq!(pos, 0);
    // Mark 4 times completed, but expect clamp at 3
    pm.inc_completed();
    pm.inc_completed();
    pm.inc_completed();
    pm.inc_completed();
    let (pos, len) = pm.total_state_for_tests().unwrap();
    assert_eq!(len, 3);
    assert_eq!(pos, 3);
}
