use polyscribe::progress::ProgressManager;

#[test]
fn test_single_mode_has_no_total_bar_and_three_bars() {
    // Use hidden backend suitable for tests
    let pm = ProgressManager::new_for_tests_single_hidden();
    // No total bar should be present
    assert!(pm.total_state_for_tests().is_none(), "single mode must not expose a total bar");
    // Bar count: header + info + current
    assert_eq!(pm.testing_bar_count(), 3);
}

#[test]
fn test_multi_mode_has_total_bar_and_four_bars() {
    let pm = ProgressManager::new_for_tests_multi_hidden(2);
    // Total bar should exist with the provided length
    let (pos, len) = pm.total_state_for_tests().expect("multi mode should expose total bar");
    assert_eq!(pos, 0);
    assert_eq!(len, 2);
    // Bar count: header + info + current + total
    assert_eq!(pm.testing_bar_count(), 4);
}
