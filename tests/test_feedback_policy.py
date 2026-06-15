from bruteforce_canvas.orchestration import (
    CandidateState,
    FeedbackPolicyError,
    apply_candidate_feedback,
)
from bruteforce_canvas.shared import FeedbackAction


def curated_candidate() -> CandidateState:
    return CandidateState(
        candidate_id="cand_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=7,
        promoted=True,
        curated=True,
    )


def test_accept_feedback_keeps_promoted_curated_and_records_positive_source():
    result = apply_candidate_feedback(curated_candidate(), FeedbackAction.ACCEPT)

    assert result.candidate.promoted is True
    assert result.candidate.curated is True
    assert result.applied is True
    assert result.learning_delta["alpha"] > 0
    assert result.signal_source == "swipe_feedback"


def test_reject_feedback_demotes_and_removes_from_curated_flow():
    result = apply_candidate_feedback(curated_candidate(), FeedbackAction.REJECT)

    assert result.candidate.promoted is False
    assert result.candidate.curated is False
    assert result.candidate.demoted is True
    assert result.learning_delta["beta"] == 1.0


def test_shred_feedback_is_stronger_negative_and_not_iqa_negative_dataset():
    result = apply_candidate_feedback(curated_candidate(), FeedbackAction.SHRED)

    assert result.candidate.promoted is False
    assert result.candidate.curated is False
    assert result.candidate.demoted is True
    assert result.learning_delta["beta"] == 2.0
    assert result.include_in_iqa_negative_dataset is False


def test_feedback_rejects_non_pre_curated_candidates():
    candidate = curated_candidate().model_copy(update={"promoted": False, "curated": False})

    try:
        apply_candidate_feedback(candidate, FeedbackAction.ACCEPT)
    except FeedbackPolicyError as error:
        assert str(error) == "feedback is only accepted for promoted and curated candidates"
    else:
        raise AssertionError("expected FeedbackPolicyError")
