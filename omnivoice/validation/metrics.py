"""Deterministic text and number-span metrics for hard-number validation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_OperationTag = Literal["equal", "substitute", "delete", "insert"]
_Operation = tuple[_OperationTag, int | None, int | None]


def normalize_ru(text: str) -> str:
    """Apply the hard-number benchmark's Russian text normalization."""
    value = text.lower().replace("ё", "е").replace("+", "")
    value = re.sub(r"[^а-я0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class EditCounts:
    """Raw Levenshtein counts over one kind of text unit."""

    substitutions: int
    deletions: int
    insertions: int
    correct: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return (
            self.errors / self.reference_units
            if self.reference_units
            else float(self.errors > 0)
        )


def _alignment(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> list[_Operation]:
    """Align reference to hypothesis with benchmark-exact traceback ordering."""
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)
    distances = [list(range(hypothesis_length + 1))]
    distances.extend(
        [index] + [0] * hypothesis_length
        for index in range(1, reference_length + 1)
    )

    for ref_index in range(1, reference_length + 1):
        ref_unit = reference[ref_index - 1]
        for hyp_index in range(1, hypothesis_length + 1):
            substitution_cost = int(ref_unit != hypothesis[hyp_index - 1])
            distances[ref_index][hyp_index] = min(
                distances[ref_index - 1][hyp_index] + 1,
                distances[ref_index][hyp_index - 1] + 1,
                distances[ref_index - 1][hyp_index - 1] + substitution_cost,
            )

    ref_index = reference_length
    hyp_index = hypothesis_length
    operations: list[_Operation] = []
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0:
            substitution_cost = int(
                reference[ref_index - 1] != hypothesis[hyp_index - 1]
            )
            if (
                distances[ref_index][hyp_index]
                == distances[ref_index - 1][hyp_index - 1] + substitution_cost
            ):
                operations.append(
                    (
                        "equal" if substitution_cost == 0 else "substitute",
                        ref_index - 1,
                        hyp_index - 1,
                    )
                )
                ref_index -= 1
                hyp_index -= 1
                continue

        if (
            ref_index > 0
            and distances[ref_index][hyp_index]
            == distances[ref_index - 1][hyp_index] + 1
        ):
            operations.append(("delete", ref_index - 1, None))
            ref_index -= 1
            continue

        operations.append(("insert", None, hyp_index - 1))
        hyp_index -= 1

    operations.reverse()
    return operations


def edit_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> EditCounts:
    """Return raw substitution, deletion, insertion, and correct counts."""
    operations = _alignment(reference, hypothesis)
    return EditCounts(
        substitutions=sum(tag == "substitute" for tag, _, _ in operations),
        deletions=sum(tag == "delete" for tag, _, _ in operations),
        insertions=sum(tag == "insert" for tag, _, _ in operations),
        correct=sum(tag == "equal" for tag, _, _ in operations),
        reference_units=len(reference),
    )


def _word_units(text: str) -> list[str]:
    return normalize_ru(text).split()


def _character_units(text: str) -> list[str]:
    return list(normalize_ru(text).replace(" ", ""))


def number_span(
    raw_text: str, gold: str
) -> tuple[list[str], int | None, int | None]:
    """Locate the contiguous gold span that verbalizes raw digits."""
    raw_words = _word_units(raw_text)
    gold_words = _word_units(gold)
    operations = _alignment(raw_words, gold_words)
    changed_gold_indices = [
        gold_index
        for tag, _, gold_index in operations
        if tag in {"substitute", "insert"} and gold_index is not None
    ]
    if not changed_gold_indices:
        return [], None, None

    start = min(changed_gold_indices)
    end = max(changed_gold_indices)
    return gold_words[start : end + 1], start, end


def hypothesis_span(
    gold: str,
    hypothesis: str,
    start: int | None,
    end: int | None,
) -> list[str]:
    """Map an inclusive gold word span into its aligned hypothesis range."""
    if start is None or end is None:
        return []

    gold_words = _word_units(gold)
    if start < 0 or end < start or end >= len(gold_words):
        raise ValueError(
            f"invalid gold span [{start}, {end}] for {len(gold_words)} words"
        )

    hypothesis_words = _word_units(hypothesis)
    mapped_indices = {
        gold_index: hypothesis_index
        for tag, gold_index, hypothesis_index in _alignment(
            gold_words, hypothesis_words
        )
        if tag in {"equal", "substitute"}
        and gold_index is not None
        and hypothesis_index is not None
    }
    corresponding = [
        mapped_indices[index]
        for index in range(start, end + 1)
        if index in mapped_indices
    ]
    if not corresponding:
        return []
    return hypothesis_words[min(corresponding) : max(corresponding) + 1]


@dataclass(frozen=True)
class UtteranceScore:
    """Normalized text, number spans, and raw edit counts for one utterance."""

    category: str
    raw_text: str
    gold: str
    hypothesis: str
    ref_number: str
    hyp_number: str
    utterance_words: EditCounts
    utterance_chars: EditCounts
    number_words: EditCounts | None
    number_chars: EditCounts | None

    @property
    def utt_wer(self) -> float:
        return self.utterance_words.rate

    @property
    def utt_cer(self) -> float:
        return self.utterance_chars.rate

    @property
    def num_wer(self) -> float | None:
        return self.number_words.rate if self.number_words is not None else None

    @property
    def num_cer(self) -> float | None:
        return self.number_chars.rate if self.number_chars is not None else None


def score_utterance(
    raw_text: str,
    gold: str,
    hypothesis: str,
    category: str = "",
) -> UtteranceScore:
    """Score full utterance and raw-digit-derived number span independently."""
    normalized_gold = normalize_ru(gold)
    normalized_hypothesis = normalize_ru(hypothesis)
    gold_words = normalized_gold.split()
    hypothesis_words = normalized_hypothesis.split()
    utterance_words = edit_counts(gold_words, hypothesis_words)
    utterance_chars = edit_counts(
        list(normalized_gold.replace(" ", "")),
        list(normalized_hypothesis.replace(" ", "")),
    )

    ref_number_tokens, start, end = number_span(raw_text, normalized_gold)
    if start is None or end is None:
        return UtteranceScore(
            category=category,
            raw_text=raw_text,
            gold=normalized_gold,
            hypothesis=normalized_hypothesis,
            ref_number="",
            hyp_number="",
            utterance_words=utterance_words,
            utterance_chars=utterance_chars,
            number_words=None,
            number_chars=None,
        )

    hypothesis_number_tokens = hypothesis_span(
        normalized_gold, normalized_hypothesis, start, end
    )
    ref_number = " ".join(ref_number_tokens)
    hyp_number = " ".join(hypothesis_number_tokens)
    return UtteranceScore(
        category=category,
        raw_text=raw_text,
        gold=normalized_gold,
        hypothesis=normalized_hypothesis,
        ref_number=ref_number,
        hyp_number=hyp_number,
        utterance_words=utterance_words,
        utterance_chars=utterance_chars,
        number_words=edit_counts(ref_number_tokens, hypothesis_number_tokens),
        number_chars=edit_counts(
            _character_units(ref_number), _character_units(hyp_number)
        ),
    )


@dataclass(frozen=True)
class AggregateBlock:
    """Micro-aggregated counts and rates for one category or all rows."""

    utterances: int
    number_spans: int
    utterance_words: EditCounts
    utterance_chars: EditCounts
    number_words: EditCounts
    number_chars: EditCounts

    @property
    def utt_wer(self) -> float:
        return self.utterance_words.rate

    @property
    def utt_cer(self) -> float:
        return self.utterance_chars.rate

    @property
    def num_wer(self) -> float:
        return self.number_words.rate

    @property
    def num_cer(self) -> float:
        return self.number_chars.rate


@dataclass(frozen=True)
class AggregateScores:
    """Overall and deterministically ordered per-category aggregate blocks."""

    overall: AggregateBlock
    per_category: dict[str, AggregateBlock]


def _sum_counts(counts: Sequence[EditCounts]) -> EditCounts:
    return EditCounts(
        substitutions=sum(item.substitutions for item in counts),
        deletions=sum(item.deletions for item in counts),
        insertions=sum(item.insertions for item in counts),
        correct=sum(item.correct for item in counts),
        reference_units=sum(item.reference_units for item in counts),
    )


def _aggregate_block(scores: Sequence[UtteranceScore]) -> AggregateBlock:
    number_word_counts = [
        score.number_words for score in scores if score.number_words is not None
    ]
    number_char_counts = [
        score.number_chars for score in scores if score.number_chars is not None
    ]
    return AggregateBlock(
        utterances=len(scores),
        number_spans=len(number_word_counts),
        utterance_words=_sum_counts(
            [score.utterance_words for score in scores]
        ),
        utterance_chars=_sum_counts(
            [score.utterance_chars for score in scores]
        ),
        number_words=_sum_counts(number_word_counts),
        number_chars=_sum_counts(number_char_counts),
    )


def aggregate_scores(scores: Sequence[UtteranceScore]) -> AggregateScores:
    """Micro-average scores overall and per category using matching units."""
    if not scores:
        raise ValueError("aggregate_scores requires at least one utterance score")

    grouped: defaultdict[str, list[UtteranceScore]] = defaultdict(list)
    for score in scores:
        grouped[score.category].append(score)
    return AggregateScores(
        overall=_aggregate_block(scores),
        per_category={
            category: _aggregate_block(grouped[category])
            for category in sorted(grouped)
        },
    )
