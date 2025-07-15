import pytest
from bitrecs.utils import constants as CONST
from bitrecs.validator.reward import measure_request_difficulty

# Constants for test
MIN_CONTEXT_LEN = 100
MAX_CONTEXT_LEN = CONST.MAX_CONTEXT_TEXT_LENGTH
MIN_RECS = CONST.MIN_RECS_PER_REQUEST
MAX_RECS = CONST.MAX_RECS_PER_REQUEST
MIN_PARTICIPANTS = 1
MAX_PARTICIPANTS = 16

@pytest.mark.parametrize(
    "context_len,num_recs,num_participants,expected_min,expected_max",
    [
        # Easiest request (all minimums)
        (MIN_CONTEXT_LEN, MIN_RECS, MIN_PARTICIPANTS, 0.9, 0.9),
        # Typical request (medium)
        (10000, 5, 8, 0.9, 1.0),
        # Hardest request (all maximums)
        (MAX_CONTEXT_LEN, MAX_RECS, MAX_PARTICIPANTS, 1.0, 1.0),
        # Over max participants (should be capped at max_decay)
        (MAX_CONTEXT_LEN, MAX_RECS, 24, 1.0, 1.0),
        # Below min context (should be capped at min_decay)
        (10, MIN_RECS, MIN_PARTICIPANTS, 0.9, 0.9),
        # Edge: min context, max recs, min participants
        (MIN_CONTEXT_LEN, MAX_RECS, MIN_PARTICIPANTS, 0.9, 0.95),
        # Edge: max context, min recs, max participants
        (MAX_CONTEXT_LEN, MIN_RECS, MAX_PARTICIPANTS, 0.95, 1.0),
        # Edge: mid context, min recs, mid participants
        (5000, MIN_RECS, 8, 0.9, 0.97),
        # Edge: mid context, max recs, mid participants
        (5000, MAX_RECS, 8, 0.9, 0.99),
    ]
)
def test_measure_request_difficulty(context_len, num_recs, num_participants, expected_min, expected_max):
    sku = "ABC123"
    context = "x" * context_len
    difficulty = measure_request_difficulty(
        sku=sku,
        context=context,
        num_recs=num_recs,
        num_participants=num_participants,
        min_context_len=MIN_CONTEXT_LEN,
        max_context_len=MAX_CONTEXT_LEN,
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

def test_difficulty_increases_with_context():
    sku = "SKU"
    num_recs = 5
    num_participants = 8
    context_short = "x" * MIN_CONTEXT_LEN
    context_long = "x" * MAX_CONTEXT_LEN
    diff_short = measure_request_difficulty(sku, context_short, num_recs, num_participants,
                                           min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                           min_recs=MIN_RECS, max_recs=MAX_RECS,
                                           min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                           min_decay=0.9, max_decay=1.0)
    diff_long = measure_request_difficulty(sku, context_long, num_recs, num_participants,
                                          min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    assert diff_long > diff_short

def test_difficulty_increases_with_num_recs():
    sku = "SKU"
    context = "x" * 1000
    num_participants = 8
    diff_few = measure_request_difficulty(sku, context, MIN_RECS, num_participants,
                                          min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    diff_many = measure_request_difficulty(sku, context, MAX_RECS, num_participants,
                                           min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                           min_recs=MIN_RECS, max_recs=MAX_RECS,
                                           min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                           min_decay=0.9, max_decay=1.0)
    assert diff_many > diff_few

def test_difficulty_increases_with_num_participants():
    sku = "SKU"
    context = "x" * 1000
    num_recs = 5
    diff_few = measure_request_difficulty(sku, context, num_recs, MIN_PARTICIPANTS,
                                          min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                          min_recs=MIN_RECS, max_recs=MAX_RECS,
                                          min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                          min_decay=0.9, max_decay=1.0)
    diff_many = measure_request_difficulty(sku, context, num_recs, MAX_PARTICIPANTS,
                                           min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                           min_recs=MIN_RECS, max_recs=MAX_RECS,
                                           min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                           min_decay=0.9, max_decay=1.0)
    assert diff_many > diff_few

def test_difficulty_is_min_decay_for_minimums():
    sku = "SKU"
    context = "x" * MIN_CONTEXT_LEN
    num_recs = MIN_RECS
    num_participants = MIN_PARTICIPANTS
    diff = measure_request_difficulty(sku, context, num_recs, num_participants,
                                      min_context_len=MIN_CONTEXT_LEN, max_context_len=MAX_CONTEXT_LEN,
                                      min_recs=MIN_RECS, max_recs=MAX_RECS,
                                      min_participants=MIN_PARTICIPANTS, max_participants=MAX_PARTICIPANTS,
                                      min_decay=0.9, max_decay=1.0)
    assert diff == 0.9, f"Expected difficulty to be 0.9 for minimums, got {diff}"