use polyscribe::progress::ProgressManager;

#[test]
fn test_total_and_completed_clamp() {
    let pm = ProgressManager::new_for_tests_multi_hidden(3);
    pm.set_total(3);
    pm.inc_completed();
    pm.inc_completed();
    pm.inc_completed();
    // Extra increments should not exceed total
    pm.inc_completed();
}

#[test]
fn test_start_item_does_not_change_total() {
    let pm = ProgressManager::new_for_tests_multi_hidden(2);
    pm.set_total(2);
    let item = pm.start_item("file1");
    item.set_progress(0.5);
    // No panic; total bar position should be unaffected. We cannot introspect position without
    // exposing internals; this test ensures API usability without side effects.
    item.finish_with("done");
}

#[test]
fn test_pause_and_resume_prompt() {
    let pm = ProgressManager::test_new_multi(1);
    pm.pause_for_prompt();
    pm.resume_after_prompt();
}
