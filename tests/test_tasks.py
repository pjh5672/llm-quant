"""Answer parsing is where a generation task silently goes wrong, so it is pinned down."""

import pytest

from llmquant.core.datasets.tasks import TASKS, _score_choice, _score_number, get_generation_task

CHOICE = {"answer": "B", "choices": ["A", "B", "C", "D"]}
NUMBER = {"answer": "18", "choices": None}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", True),
        ("B.", True),
        ("The answer is B.", True),
        ("**B**", True),
        ("b", True),  # a lowercase answer still counts
        ("A", False),
        ("The answer is A.", False),
        ("I am not sure.", False),
        # the English article must not be read as option A
        ("a careful reading shows B is correct", True),
        ("a careful reading shows A is correct", False),
    ],
)
def test_choice_scoring(text, expected):
    assert _score_choice(text, CHOICE) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### 18", True),
        ("so the total is 18\n#### 18", True),
        ("blah 9 then 2 more\n#### 18", True),
        ("the answer is 18", True),  # no marker: fall back to the last number
        ("#### 1,800", False),
        ("#### 19", False),
        ("no numbers here", False),
        ("she had 18 eggs but sold 12\n#### 12", False),  # the marker wins over an earlier match
    ],
)
def test_number_scoring(text, expected):
    assert _score_number(text, NUMBER) is expected


def test_number_scoring_ignores_thousands_separators():
    assert _score_number("#### 1,800", {"answer": "1800", "choices": None}) is True


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown generation task"):
        get_generation_task("nope")


@pytest.mark.parametrize("name", sorted(TASKS))
def test_each_task_loads_and_is_well_formed(name):
    examples, spec = get_generation_task(name, limit=3)
    assert len(examples) == 3
    assert all(e["prompt"] and e["answer"] for e in examples)
    assert spec["max_new_tokens"] > 0
    # the reference answer must score as correct against itself
    reference = examples[0]["answer"]
    text = reference if name == "arc_easy" else f"#### {reference}"
    assert spec["score"](text, examples[0]) is True
