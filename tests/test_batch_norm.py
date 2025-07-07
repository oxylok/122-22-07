import unittest
import numpy as np
from bitrecs.validator.reward import calculate_percentile_timing_penalty, ALPHA_TIME_DECAY

class TestCalculatePercentileTimingPenalty(unittest.TestCase):
    
    def test_single_time_returns_default_penalty(self):
        """Test that single time returns default penalty"""
        result = calculate_percentile_timing_penalty(2.0, [2.0], 123)
        expected = ALPHA_TIME_DECAY * 0.5
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_fastest_miner_minimal_penalty(self):
        """Test that fastest miner gets minimal penalty"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_percentile_timing_penalty(1.0, all_times, 123)
        # count_below = 0, count_equal = 1
        # rank = 0 + (1 + 1) / 2 = 1.0
        # percentile = 1.0 / 5 = 0.2
        expected = ALPHA_TIME_DECAY * 0.1 * 0.2  # Top 50% penalty
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_slowest_miner_max_penalty(self):
        """Test that slowest miner gets maximum penalty"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_percentile_timing_penalty(5.0, all_times, 123)
        # count_below = 4, count_equal = 1
        # rank = 4 + (1 + 1) / 2 = 5.0
        # percentile = 5.0 / 5 = 1.0
        # Bottom 50%: (1.0 - 0.5) * 2 = 0.5 * 2 = 1.0
        # penalty = ALPHA_TIME_DECAY * (0.05 + 0.95 * 1.0) = ALPHA_TIME_DECAY * 1.0
        expected = ALPHA_TIME_DECAY * 1.0
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_median_miner_boundary(self):
        """Test miner at 50th percentile (boundary between penalty curves)"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_percentile_timing_penalty(3.0, all_times, 123)
        # count_below = 2, count_equal = 1
        # rank = 2 + (1 + 1) / 2 = 3.0
        # percentile = 3.0 / 5 = 0.6
        # Bottom 50%: penalty = ALPHA_TIME_DECAY * (0.05 + 0.95 * (0.6 - 0.5) * 2)
        percentile = 0.6
        expected = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile - 0.5) * 2)
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_duplicate_times_average_rank(self):
        """Test that duplicate times get average rank"""
        all_times = [1.0, 2.0, 2.0, 2.0, 5.0]
        result = calculate_percentile_timing_penalty(2.0, all_times, 123)
        # count_below = 1, count_equal = 3
        # rank = 1 + (3 + 1) / 2 = 3.0
        # percentile = 3.0 / 5 = 0.6
        percentile = 3.0 / 5.0
        # Bottom 50%: penalty = ALPHA_TIME_DECAY * (0.05 + 0.95 * (0.6 - 0.5) * 2)
        expected = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile - 0.5) * 2)
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_penalty_curve_transition(self):
        """Test penalty curve transition from top 50% to bottom 50%"""
        all_times = [1.0, 2.0, 3.0, 4.0]
        
        # Test 37.5th percentile (top 50%)
        result_25 = calculate_percentile_timing_penalty(2.0, all_times, 123)
        # count_below = 1, count_equal = 1
        # rank = 1 + (1 + 1) / 2 = 2.0
        # percentile = 2.0 / 4 = 0.5
        expected_25 = ALPHA_TIME_DECAY * 0.1 * 0.5  # Top 50% penalty
        self.assertAlmostEqual(result_25, expected_25, places=4)
        
        # Test 75th percentile (bottom 50%)
        result_75 = calculate_percentile_timing_penalty(4.0, all_times, 123)
        # count_below = 3, count_equal = 1
        # rank = 3 + (1 + 1) / 2 = 4.0
        # percentile = 4.0 / 4 = 1.0
        percentile_75 = 1.0
        expected_75 = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile_75 - 0.5) * 2)
        self.assertAlmostEqual(result_75, expected_75, places=4)
        
        # Bottom 50% penalty should be higher than top 50%
        self.assertGreater(result_75, result_25)
    
    def test_exact_50th_percentile(self):
        """Test behavior exactly at 50th percentile"""
        all_times = [1.0, 2.0, 3.0, 4.0]
        result = calculate_percentile_timing_penalty(2.5, all_times, 123)
        # 2.5 is not in the list, but count_below = 2, count_equal = 0
        # rank = 2 + (0 + 1) / 2 = 2.5
        # percentile = 2.5 / 4 = 0.625
        percentile = 0.625
        # Bottom 50%: penalty = ALPHA_TIME_DECAY * (0.05 + 0.95 * (0.625 - 0.5) * 2)
        expected = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile - 0.5) * 2)
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_empty_times_list(self):
        """Test edge case with empty times list"""
        result = calculate_percentile_timing_penalty(2.0, [], 123)
        expected = ALPHA_TIME_DECAY * 0.5
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_penalty_non_negative(self):
        """Test that penalty is always non-negative"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for time in all_times:
            result = calculate_percentile_timing_penalty(time, all_times, 123)
            self.assertGreaterEqual(result, 0.0)
    
    def test_penalty_bounded_by_alpha(self):
        """Test that penalty never exceeds ALPHA_TIME_DECAY"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for time in all_times:
            result = calculate_percentile_timing_penalty(time, all_times, 123)
            self.assertLessEqual(result, ALPHA_TIME_DECAY)
    
    def test_rank_calculation_examples(self):
        """Test specific rank calculations to verify the formula"""
        all_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test each time and verify rank calculation
        test_cases = [
            (1.0, 1.0),  # count_below=0, count_equal=1, rank=0+(1+1)/2=1.0
            (2.0, 2.0),  # count_below=1, count_equal=1, rank=1+(1+1)/2=2.0
            (3.0, 3.0),  # count_below=2, count_equal=1, rank=2+(1+1)/2=3.0
            (4.0, 4.0),  # count_below=3, count_equal=1, rank=3+(1+1)/2=4.0
            (5.0, 5.0),  # count_below=4, count_equal=1, rank=4+(1+1)/2=5.0
        ]
        
        for axon_time, expected_rank in test_cases:
            count_below = sum(1 for t in all_times if t < axon_time)
            count_equal = sum(1 for t in all_times if t == axon_time)
            calculated_rank = count_below + (count_equal + 1) / 2
            self.assertEqual(calculated_rank, expected_rank, 
                           f"Rank calculation failed for time {axon_time}")

if __name__ == '__main__':
    # Run the tests
    unittest.main()