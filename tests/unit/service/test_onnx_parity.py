import pytest

from core.onnx_parity import evaluate_backend_parity


def test_onnx_candidate_passes_only_with_parity_and_an_improvement():
    labels = ["benign"] * 500 + ["attack"] * 500
    reference = ["allow"] * 500 + ["block"] * 500
    candidate = list(reference)

    report = evaluate_backend_parity(
        reference,
        candidate,
        labels,
        reference_p95_ms=100,
        candidate_p95_ms=60,
    )

    assert report["verdict_disagreement_rate"] == 0
    assert report["passed"] is True


def test_onnx_candidate_fails_over_half_percent_disagreement():
    labels = ["benign"] * 500 + ["attack"] * 500
    reference = ["allow"] * 500 + ["block"] * 500
    candidate = list(reference)
    candidate[:6] = ["flag"] * 6

    report = evaluate_backend_parity(
        reference,
        candidate,
        labels,
        reference_p95_ms=100,
        candidate_p95_ms=50,
    )

    assert report["verdict_disagreement_rate"] == pytest.approx(0.006)
    assert report["passed"] is False


def test_onnx_candidate_fails_a_false_block_regression():
    labels = ["benign"] * 500 + ["attack"] * 500
    reference = ["allow"] * 500 + ["block"] * 500
    candidate = list(reference)
    candidate[0] = "block"

    report = evaluate_backend_parity(
        reference,
        candidate,
        labels,
        reference_p95_ms=100,
        candidate_p95_ms=50,
    )

    assert report["candidate_false_block_rate"] > report["reference_false_block_rate"]
    assert report["passed"] is False
