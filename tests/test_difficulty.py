import pytest
from bitrecs.utils import constants as CONST
from bitrecs.validator.reward import measure_request_difficulty


MIN_CATALOG_SIZE = CONST.MIN_CATALOG_SIZE
MAX_CATALOG_SIZE = CONST.MAX_CATALOG_SIZE
MIN_RECS = CONST.MIN_RECS_PER_REQUEST
MAX_RECS = CONST.MAX_RECS_PER_REQUEST
MIN_PARTICIPANTS = 1
MAX_PARTICIPANTS = 16


def color_code_difficulty(difficulty: float) -> str:
    if difficulty <= 0.93:
        return "green"
    elif difficulty <= 0.97:
        return "yellow"
    else:
        return "red"



@pytest.mark.parametrize(
    "catalog_size,num_recs,num_participants,expected_min,expected_max",
    [
        # Easiest request (all minimums)
        (MIN_CATALOG_SIZE, MIN_RECS, MIN_PARTICIPANTS, 0.9, 0.9),
        # Typical request (medium)
        (500, 5, 8, 0.9, 1.0),
        # Hardest request (all maximums)
        (MAX_CATALOG_SIZE, MAX_RECS, MAX_PARTICIPANTS, 1.0, 1.0),
        # Over max participants (should be capped at max_decay)
        (MAX_CATALOG_SIZE, MAX_RECS, 24, 1.0, 1.0),
        # Below min catalog size (should be capped at min_decay)
        (1, MIN_RECS, MIN_PARTICIPANTS, 0.9, 0.9),
        # Edge: min catalog, max recs, min participants
        (MIN_CATALOG_SIZE, MAX_RECS, MIN_PARTICIPANTS, 0.9, 0.95),
        # Edge: max catalog, min recs, max participants
        (MAX_CATALOG_SIZE, MIN_RECS, MAX_PARTICIPANTS, 0.95, 1.0),
        # Edge: mid catalog, min recs, mid participants
        (500, MIN_RECS, 8, 0.9, 0.97),
        # Edge: mid catalog, max recs, mid participants
        (500, MAX_RECS, 8, 0.9, 0.99),
    ]
)
def test_measure_request_difficulty(catalog_size, num_recs, num_participants, expected_min, expected_max):
    sku = "ABC123"
    difficulty = measure_request_difficulty(
        sku=sku,
        catalog_size=catalog_size,
        num_recs=num_recs,
        num_participants=num_participants,
        min_catalog_size=MIN_CATALOG_SIZE,
        max_catalog_size=MAX_CATALOG_SIZE,
        min_recs=MIN_RECS,
        max_recs=MAX_RECS,
        min_participants=MIN_PARTICIPANTS,
        max_participants=MAX_PARTICIPANTS,
        base=1.0,
        min_decay=0.9,
        max_decay=1.0
    )
    # Difficulty should always be in [0.9, 1.0]
    assert 0.9 <= difficulty <= 1.0
    # Should be within the expected range for this scenario
    assert expected_min <= difficulty <= expected_max

def test_difficulty_increases_with_catalog_size():
    sku = "SKU"
    num_recs = 5
    num_participants = 8
    catalog_small = MIN_CATALOG_SIZE
    catalog_large = MAX_CATALOG_SIZE
    diff_small = measure_request_difficulty(sku, catalog_small, num_recs, num_participants,
                                           min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                           min_recs=MIN_RECS, max_recs=MAX_RECS,
                                           min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                           min_decay=0.9, max_decay=1.0)
    diff_large = measure_request_difficulty(sku, catalog_large, num_recs, num_participants,
                                           min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                           min_recs=MIN_RECS, max_recs=MAX_RECS,
                                           min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                           min_decay=0.9, max_decay=1.0)
    assert diff_large > diff_small

def test_difficulty_increases_with_num_recs():
    sku = "SKU"
    catalog_size = 500
    num_participants = 8
    diff_few = measure_request_difficulty(sku, catalog_size, MIN_RECS, num_participants,
                                          min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    diff_many = measure_request_difficulty(sku, catalog_size, MAX_RECS, num_participants,
                                          min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    assert diff_many > diff_few

def test_difficulty_increases_with_num_participants():
    sku = "SKU"
    catalog_size = 500
    num_recs = 5
    diff_few = measure_request_difficulty(sku, catalog_size, num_recs, MIN_PARTICIPANTS,
                                          min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    diff_many = measure_request_difficulty(sku, catalog_size, num_recs, MAX_PARTICIPANTS,
                                          min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    assert diff_many > diff_few

def test_difficulty_is_min_decay_for_minimums():
    sku = "SKU"
    catalog_size = MIN_CATALOG_SIZE
    num_recs = MIN_RECS
    num_participants = MIN_PARTICIPANTS
    diff = measure_request_difficulty(sku, catalog_size, num_recs, num_participants,
                                      min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
                                      min_recs=MIN_RECS, max_recs=MAX_RECS,
                                      min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                      min_decay=0.9, max_decay=1.0)
    assert diff == 0.9, f"Expected difficulty to be 0.9 for minimums, got {diff}"

@pytest.mark.parametrize(
    "difficulty,expected_color",
    [
        (0.9, "green"),
        (0.92, "green"),
        (0.94, "yellow"),
        (0.96, "yellow"),
        (0.98, "red"),
        (1.0, "red"),
    ]
)
def test_color_code_difficulty(difficulty, expected_color):
    color = color_code_difficulty(difficulty)
    assert color == expected_color, f"For difficulty {difficulty}, expected {expected_color} but got {color}"

@pytest.mark.parametrize(
    "difficulty,expected_color",
    [
        (0.93, "green"),    # boundary
        (0.97, "yellow"),   # boundary
        (0.99, "red"),      # above boundary
    ]
)
def test_color_code_difficulty_boundaries(difficulty, expected_color):
    color = color_code_difficulty(difficulty)
    assert color == expected_color

def test_measure_request_difficulty_negative_inputs():
    sku = "SKU"
    diff = measure_request_difficulty(
        sku, -10, -1, -5,
        min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
        min_recs=MIN_RECS, max_recs=MAX_RECS,
        min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
        min_decay=0.9, max_decay=1.0
    )
    assert diff == 0.9  # Should clamp to min_decay

def test_measure_request_difficulty_large_inputs():
    sku = "SKU"
    diff = measure_request_difficulty(
        sku, 10000, 1000, 1000,
        min_catalog_size=MIN_CATALOG_SIZE, max_catalog_size=MAX_CATALOG_SIZE,
        min_recs=MIN_RECS, max_recs=MAX_RECS,
        min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
        min_decay=0.9, max_decay=1.0
    )
    assert diff <= 1.0 and diff > 0.9  # Should be in the valid range