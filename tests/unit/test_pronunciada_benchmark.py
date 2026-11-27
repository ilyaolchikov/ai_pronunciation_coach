from pronunciation_coach.pronunciada_benchmark import (
    classification_metrics,
    word_index_mapping,
)


def test_classification_metrics_count_word_level_outcomes() -> None:
    metrics = classification_metrics(
        [
            {
                "words": ["one", "two", "three"],
                "truth_word_indexes": [1],
                "predicted": [1, 2],
            },
            {
                "words": ["four", "five"],
                "truth_word_indexes": [0],
                "predicted": [],
            },
        ],
        "predicted",
    )

    assert metrics == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 1 / 3,
    }


def test_word_index_mapping_handles_annotation_insertions_and_spelling_changes() -> None:
    mapping, cost = word_index_mapping(
        ["that's", "why", "sp", "its", "colour"],
        ["That's", "why", "its", "color"],
    )

    assert mapping == [0, 1, None, 2, 3]
    assert cost == 2
