use super::*;

#[test]
fn failed_final_checkpoint_overrides_bullets_completed_run()
{
    let error = resolve_outcome(Ok(()), Some(anyhow::anyhow!("checkpoint disk full")), None, false, true).unwrap_err();

    assert_eq!(error.to_string(), "checkpoint disk full");
}

#[test]
fn cancellation_is_distinct_from_success_and_preserves_real_failures()
{
    let early = || {
        TrainingError::DataLoadingError(DataLoadingError::Message(
            "Data loader ended before all scheduled batches completed".into(),
        ))
    };

    assert_eq!(resolve_outcome(Err(early()), None, None, false, true).unwrap(), Outcome::Cancelled);
    assert!(resolve_outcome(Err(early()), None, None, false, false).is_err());
    assert_eq!(resolve_outcome(Ok(()), None, None, false, false).unwrap(), Outcome::Success);
    assert_eq!(resolve_outcome(Ok(()), None, None, false, true).unwrap(), Outcome::Cancelled);
    assert!(resolve_outcome(Err(TrainingError::InvalidSchedule), None, None, false, true).is_err());
    assert!(resolve_outcome(Ok(()), None, Some("invalid record".into()), false, true).is_err());
    assert!(resolve_outcome(Ok(()), None, None, true, true).is_err());
}

#[test]
fn training_diagnostics_retain_underlying_cause()
{
    let error = training_error(TrainingError::DataLoadingError(DataLoadingError::Message("record 44 corrupt".into())));

    assert!(error.to_string().contains("train --check"));
    assert!(format!("{error:#}").contains("record 44 corrupt"));
}
