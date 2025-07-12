import unittest
import numpy as np
from bitrecs.validator.reward import calculate_percentile_timing_penalty, ALPHA_TIME_DECAY

class TestCalculatePercentileTimingPenalty(unittest.TestCase):
    
    def test_single_time_returns_default_penalty(self):
        """Test that single time returns default penalty"""
        result = calculate_percentile_timing_penalty(2.0, [2.0], "miner_123")
        expected = ALPHA_TIME_DECAY * 0.5
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_realistic_5_miner_cohort(self):
        """Test realistic 5-miner timing distribution after P_LIMIT elimination"""
        # Realistic post-elimination times: 1.2s - 3.5s range
        all_times = [1.2, 1.8, 2.3, 2.9, 3.5]
        
        # Test fastest miner (1.2s) - 20th percentile
        result_fastest = calculate_percentile_timing_penalty(1.2, all_times, "miner_fast")
        # rank = 0 + (1 + 1) / 2 = 1.0, percentile = 1.0 / 5 = 0.2
        expected_fastest = ALPHA_TIME_DECAY * 0.1 * 0.2
        self.assertAlmostEqual(result_fastest, expected_fastest, places=4)
        
        # Test median miner (2.3s) - 60th percentile  
        result_median = calculate_percentile_timing_penalty(2.3, all_times, "miner_median")
        # rank = 2 + (1 + 1) / 2 = 3.0, percentile = 3.0 / 5 = 0.6
        expected_median = ALPHA_TIME_DECAY * (0.05 + 0.95 * (0.6 - 0.5) * 2)
        self.assertAlmostEqual(result_median, expected_median, places=4)
        
        # Test slowest miner (3.5s) - 100th percentile
        result_slowest = calculate_percentile_timing_penalty(3.5, all_times, "miner_slow")
        # rank = 4 + (1 + 1) / 2 = 5.0, percentile = 5.0 / 5 = 1.0
        expected_slowest = ALPHA_TIME_DECAY * (0.05 + 0.95 * (1.0 - 0.5) * 2)
        self.assertAlmostEqual(result_slowest, expected_slowest, places=4)
        
        # Verify penalty progression
        self.assertLess(result_fastest, result_median)
        self.assertLess(result_median, result_slowest)
    
    def test_realistic_8_miner_cohort_mixed_performance(self):
        """Test realistic 8-miner cohort with mixed LLM vs caching performance"""
        # Mixed timing: fast cache (1.1-1.5s), medium LLM (1.8-2.5s), slow LLM (3.0-4.2s)
        all_times = [1.15, 1.42, 1.85, 2.1, 2.4, 2.8, 3.2, 4.1]
        
        test_cases = [
            (1.15, 1.0, "fast_cache"),      # 12.5th percentile
            (1.85, 3.0, "medium_llm"),      # 37.5th percentile  
            (2.4, 5.0, "median_llm"),       # 62.5th percentile
            (4.1, 8.0, "slow_llm")          # 100th percentile
        ]
        
        results = []
        for axon_time, expected_rank, label in test_cases:
            result = calculate_percentile_timing_penalty(axon_time, all_times, f"miner_{label}")
            percentile = expected_rank / len(all_times)
            
            if percentile <= 0.5:
                expected = ALPHA_TIME_DECAY * 0.1 * percentile
            else:
                expected = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile - 0.5) * 2)
            
            self.assertAlmostEqual(result, expected, places=4, 
                                 msg=f"Failed for {label} at {axon_time}s")
            results.append(result)
        
        # Verify monotonic increase in penalties
        for i in range(len(results) - 1):
            self.assertLess(results[i], results[i + 1])
    
    def test_realistic_12_miner_large_cohort(self):
        """Test realistic 12-miner large cohort with statistical outliers"""
        # Large cohort with potential outliers: 1.1s - 4.7s spread
        all_times = [1.12, 1.34, 1.58, 1.89, 2.15, 2.43, 2.71, 3.05, 3.38, 3.92, 4.21, 4.67]
        
        # Test key percentiles
        test_scenarios = [
            (1.12, "fastest_possible"),     # 8.3rd percentile
            (1.89, "good_performance"),     # 33.3rd percentile
            (2.71, "median_performance"),   # 58.3rd percentile  
            (3.92, "slow_performance"),     # 83.3rd percentile
            (4.67, "slowest_outlier")       # 100th percentile
        ]
        
        penalties = []
        for axon_time, scenario in test_scenarios:
            result = calculate_percentile_timing_penalty(axon_time, all_times, f"miner_{scenario}")
            penalties.append(result)
            
            # Verify penalty is within valid bounds
            self.assertLessEqual(result, ALPHA_TIME_DECAY)
            self.assertGreaterEqual(result, 0.0)
        
        # Verify smooth penalty progression across large cohort
        for i in range(len(penalties) - 1):
            self.assertLessEqual(penalties[i], penalties[i + 1])  # Allow equal values
        
        # Check that there's meaningful penalty differentiation
        fastest_penalty = penalties[0]
        slowest_penalty = penalties[-1]
        
        # Ensure there's at least some penalty difference
        self.assertGreater(slowest_penalty, fastest_penalty)
        
        # For large cohorts, penalty range should be meaningful
        penalty_range = slowest_penalty - fastest_penalty
        self.assertGreater(penalty_range, ALPHA_TIME_DECAY * 0.1, 
                          "Penalty range too small for large cohort differentiation")
    
    def test_duplicate_times_realistic_clustering(self):
        """Test realistic scenario with miners clustered at similar times"""
        # Realistic clustering: multiple miners with similar LLM inference times
        all_times = [1.2, 1.8, 2.1, 2.1, 2.1, 2.4, 2.7, 3.1]
        
        # Test miner in the cluster (2.1s) - should get average rank
        result = calculate_percentile_timing_penalty(2.1, all_times, "miner_clustered")
        # count_below = 2, count_equal = 3
        # rank = 2 + (3 + 1) / 2 = 4.0
        # percentile = 4.0 / 8 = 0.5
        expected_percentile = 0.5
        expected = ALPHA_TIME_DECAY * 0.1 * expected_percentile  # Exactly at boundary
        self.assertAlmostEqual(result, expected, places=4)
        
        # Test miner just outside cluster
        result_outside = calculate_percentile_timing_penalty(2.4, all_times, "miner_outside")
        # Should get higher penalty than clustered miners
        self.assertGreater(result_outside, result)
    
    def test_edge_case_small_cohort_after_eliminations(self):
        """Test small cohort after heavy P_LIMIT eliminations"""
        # Only 5 miners survived elimination out of original 12
        all_times = [1.25, 1.67, 2.1, 2.8, 3.4]
        
        penalties = []
        for i, axon_time in enumerate(all_times):
            result = calculate_percentile_timing_penalty(axon_time, all_times, f"survivor_{i}")
            penalties.append(result)
            
            # Even in small cohort, penalties should be meaningful
            self.assertGreater(result, 0.0)
            
            # Penalties should be within the valid range
            self.assertLessEqual(result, ALPHA_TIME_DECAY)
        
        # Check penalty progression
        for i in range(len(penalties) - 1):
            self.assertLessEqual(penalties[i], penalties[i + 1])
        
        # Fastest should get minimal penalty relative to max
        self.assertLess(penalties[0], ALPHA_TIME_DECAY * 0.6)
    
    def test_realistic_timing_spread_scenarios(self):
        """Test various realistic timing spread scenarios"""
        
        # Scenario 1: Tight competition (all fast LLM inference)
        tight_times = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        fastest_tight = calculate_percentile_timing_penalty(1.2, tight_times, "tight_fastest")
        slowest_tight = calculate_percentile_timing_penalty(1.7, tight_times, "tight_slowest")
        
        # Scenario 2: Wide spread (mix of caching and slow inference)  
        wide_times = [1.1, 1.8, 2.5, 3.2, 3.9, 4.6]
        fastest_wide = calculate_percentile_timing_penalty(1.1, wide_times, "wide_fastest")
        slowest_wide = calculate_percentile_timing_penalty(4.6, wide_times, "wide_slowest")
        
        # Calculate penalty spreads
        tight_penalty_spread = slowest_tight - fastest_tight
        wide_penalty_spread = slowest_wide - fastest_wide
        
        # Wide spread should have larger or equal penalty difference
        # (Due to percentile formula, might be equal in some cases)
        self.assertGreaterEqual(wide_penalty_spread, tight_penalty_spread * 0.9,
                               "Wide timing spread should create meaningful penalty differences")
        
        # Both scenarios should create competitive differentiation
        self.assertGreater(tight_penalty_spread, 0.0)
        self.assertGreater(wide_penalty_spread, 0.0)
    
    def test_boundary_conditions_realistic_ranges(self):
        """Test boundary conditions with realistic mining timing ranges"""
        
        # Test just above P_LIMIT threshold (1.01s)
        post_elimination_times = [1.02, 1.15, 1.8, 2.3, 2.9]
        result_just_above = calculate_percentile_timing_penalty(1.02, post_elimination_times, "just_above_limit")
        
        # Should get minimal penalty as fastest legitimate miner
        self.assertLess(result_just_above, ALPHA_TIME_DECAY * 0.15)
        
        # Test near timeout (5.0s limit)
        near_timeout_times = [1.5, 2.1, 2.8, 3.5, 4.9]
        result_near_timeout = calculate_percentile_timing_penalty(4.9, near_timeout_times, "near_timeout")
        
        # Should get significant penalty but within bounds
        self.assertGreater(result_near_timeout, ALPHA_TIME_DECAY * 0.5)
        self.assertLessEqual(result_near_timeout, ALPHA_TIME_DECAY)  # Can equal max penalty
    
    def test_production_cohort_distribution(self):
        """Test realistic production distribution patterns"""
        
        # Production-like distribution: 
        # - 20% fast miners (1.1-1.6s) - optimized/caching
        # - 50% normal miners (1.8-2.8s) - standard LLM  
        # - 30% slow miners (3.0-4.5s) - complex processing/slow hardware
        production_times = [
            1.12, 1.45,                    # Fast (20%)
            1.85, 2.1, 2.3, 2.6, 2.75,    # Normal (50%) 
            3.1, 3.8, 4.2                  # Slow (30%)
        ]
        
        # Test each category gets appropriate relative penalties
        fast_penalty = calculate_percentile_timing_penalty(1.12, production_times, "fast_prod")
        normal_penalty = calculate_percentile_timing_penalty(2.3, production_times, "normal_prod")  
        slow_penalty = calculate_percentile_timing_penalty(4.2, production_times, "slow_prod")
        
        # Verify production-realistic penalty progression
        self.assertLess(fast_penalty, ALPHA_TIME_DECAY * 0.2)      # Fast miners get minimal penalty
        self.assertLess(normal_penalty, ALPHA_TIME_DECAY * 0.7)    # Normal miners moderate penalty
        self.assertGreater(slow_penalty, ALPHA_TIME_DECAY * 0.6)   # Slow miners significant penalty
        
        # All penalties should be within valid bounds
        self.assertLessEqual(slow_penalty, ALPHA_TIME_DECAY)
        
        # Verify progression
        self.assertLess(fast_penalty, normal_penalty)
        self.assertLess(normal_penalty, slow_penalty)
    
    def test_penalty_formula_edge_cases(self):
        """Test edge cases of the penalty formula"""
        
        # Test minimum cohort size for competition
        min_times = [1.5, 2.0]
        result_min = calculate_percentile_timing_penalty(2.0, min_times, "min_test")
        # Should use the 100th percentile formula
        expected_min = ALPHA_TIME_DECAY * (0.05 + 0.95 * (1.0 - 0.5) * 2)
        self.assertAlmostEqual(result_min, expected_min, places=4)
        
        # Test exactly at 50th percentile boundary
        boundary_times = [1.0, 2.0, 3.0, 4.0]
        result_boundary = calculate_percentile_timing_penalty(2.5, boundary_times, "boundary_test")
        # rank = 2 + (0 + 1) / 2 = 2.5, percentile = 2.5 / 4 = 0.625
        expected_boundary = ALPHA_TIME_DECAY * (0.05 + 0.95 * (0.625 - 0.5) * 2)
        self.assertAlmostEqual(result_boundary, expected_boundary, places=4)


    def test_penalty_with_single_time(self):
        """Test penalty calculation with a single time in the cohort"""
        single_time = [2.0]
        result = calculate_percentile_timing_penalty(2.0, single_time, "single_miner")
        
        # Should return the default penalty for a single miner
        expected = ALPHA_TIME_DECAY * 0.5
        self.assertAlmostEqual(result, expected, places=4)
        

if __name__ == '__main__':
    # Run the tests with more verbose output
    unittest.main(verbosity=2)