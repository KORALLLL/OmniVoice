from __future__ import annotations

import pytest

from omnivoice.validation.metrics import (
    EditCounts,
    aggregate_scores,
    edit_counts,
    hypothesis_span,
    normalize_ru,
    number_span,
    score_utterance,
)


def test_normalize_ru_matches_benchmark_rules() -> None:
    assert normalize_ru("Ёж +ёл: 12!") == "еж ел 12"


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (("а", "б"), ("а", "в"), EditCounts(1, 0, 0, 1, 2)),
        (("а", "б"), ("а",), EditCounts(0, 1, 0, 1, 2)),
        (("а", "б"), ("а", "в", "б"), EditCounts(0, 0, 1, 2, 2)),
        (("а", "б"), (), EditCounts(0, 2, 0, 0, 2)),
        ((), ("а",), EditCounts(0, 0, 1, 0, 0)),
    ],
)
def test_edit_counts_covers_each_operation_and_empty_sequences(
    reference: tuple[str, ...],
    hypothesis: tuple[str, ...],
    expected: EditCounts,
) -> None:
    counts = edit_counts(reference, hypothesis)

    assert counts == expected
    assert counts.rate == (
        counts.errors / counts.reference_units
        if counts.reference_units
        else float(counts.errors > 0)
    )


def test_alignment_prefers_diagonal_when_substitutions_tie_with_delete_insert() -> None:
    assert edit_counts(("а", "б"), ("б", "а")) == EditCounts(
        substitutions=2,
        deletions=0,
        insertions=0,
        correct=0,
        reference_units=2,
    )


def test_alignment_prefers_deletion_before_insertion() -> None:
    ref_tokens, start, end = number_span("а б а", "б а б")

    assert ref_tokens == ["б"]
    assert (start, end) == (0, 0)


def test_number_span_maps_gold_into_hypothesis() -> None:
    score = score_utterance(
        raw_text="У меня 21 книга",
        gold="у меня двадцать одна книга",
        hypothesis="у меня двадцать две книги",
    )

    assert score.ref_number == "двадцать одна"
    assert score.hyp_number == "двадцать две"


def test_hypothesis_span_keeps_insertions_between_mapped_number_words() -> None:
    ref_tokens, start, end = number_span(
        "у меня 21 книга", "у меня двадцать одна книга"
    )

    mapped = hypothesis_span(
        "у меня двадцать одна книга",
        "у меня двадцать совсем одна книга",
        start,
        end,
    )

    assert ref_tokens == ["двадцать", "одна"]
    assert mapped == ["двадцать", "совсем", "одна"]


def test_number_span_maps_deletions_and_empty_hypothesis() -> None:
    deletion = score_utterance(
        raw_text="у меня 21 книга",
        gold="у меня двадцать одна книга",
        hypothesis="у меня двадцать книга",
    )
    empty = score_utterance(
        raw_text="у меня 21 книга",
        gold="у меня двадцать одна книга",
        hypothesis="",
    )

    assert deletion.hyp_number == "двадцать"
    assert deletion.number_words == EditCounts(0, 1, 0, 1, 2)
    assert empty.hyp_number == ""
    assert empty.number_words == EditCounts(0, 2, 0, 0, 2)
    assert empty.utterance_words == EditCounts(0, 5, 0, 0, 5)


def test_missing_number_span_is_explicit_and_excluded_from_number_metrics() -> None:
    score = score_utterance(
        raw_text="текст без замен",
        gold="текст без замен",
        hypothesis="текст без замен",
        category="missing",
    )

    aggregate = aggregate_scores([score])

    assert score.ref_number == ""
    assert score.hyp_number == ""
    assert score.number_words is None
    assert score.number_chars is None
    assert aggregate.overall.number_spans == 0
    assert aggregate.overall.num_wer == 0.0
    assert aggregate.overall.num_cer == 0.0


def test_score_returns_raw_counts_for_all_four_metric_families() -> None:
    score = score_utterance(
        raw_text="1",
        gold="кот",
        hypothesis="кит",
        category="integer",
    )

    assert score.utterance_words == EditCounts(1, 0, 0, 0, 1)
    assert score.utterance_chars == EditCounts(1, 0, 0, 2, 3)
    assert score.number_words == EditCounts(1, 0, 0, 0, 1)
    assert score.number_chars == EditCounts(1, 0, 0, 2, 3)
    assert score.utt_wer == 1.0
    assert score.utt_cer == pytest.approx(1 / 3)
    assert score.num_wer == 1.0
    assert score.num_cer == pytest.approx(1 / 3)


def test_aggregate_uses_true_character_micro_denominator_and_categories() -> None:
    long_word = score_utterance(
        raw_text="1",
        gold="абвгдежзий",
        hypothesis="абвгдежзиы",
        category="long",
    )
    ten_short_words = score_utterance(
        raw_text="а б в г д е ж з и 2",
        gold="а б в г д е ж з и й",
        hypothesis="а б в г д е ж з и й",
        category="short",
    )

    aggregate = aggregate_scores([long_word, ten_short_words])

    assert aggregate.overall.utterance_chars.errors == 1
    assert aggregate.overall.utterance_chars.reference_units == 20
    assert aggregate.overall.utt_cer == pytest.approx(1 / 20)
    assert aggregate.overall.utt_cer != pytest.approx((0.1 * 1 + 0.0 * 10) / 11)
    assert aggregate.overall.utt_wer == pytest.approx(1 / 11)
    assert list(aggregate.per_category) == ["long", "short"]
    assert aggregate.per_category["long"].utt_cer == pytest.approx(0.1)
    assert aggregate.per_category["short"].utt_cer == 0.0


def test_aggregate_sums_raw_counts_independently_for_each_metric_family() -> None:
    substitution = score_utterance("1", "кот", "кит", category="mixed")
    insertion = score_utterance("2", "дом", "домик", category="mixed")

    block = aggregate_scores([substitution, insertion]).overall

    assert block.utterance_words == EditCounts(2, 0, 0, 0, 2)
    assert block.utterance_chars == EditCounts(1, 0, 2, 5, 6)
    assert block.number_words == EditCounts(2, 0, 0, 0, 2)
    assert block.number_chars == EditCounts(1, 0, 2, 5, 6)


def test_aggregate_requires_at_least_one_score() -> None:
    with pytest.raises(ValueError, match="at least one utterance score"):
        aggregate_scores([])
