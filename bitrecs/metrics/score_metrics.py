import traceback
import bittensor as bt
import numpy as np
from bitrecs.utils import epoch

def display_normalized_analysis(validator_instance):
    """Display normalized scores that are actually used for weights"""
    try:
        normalized_scores = validator_instance.get_normalized_scores()
        
        bt.logging.info(f"\033[1;36m=== NORMALIZED WEIGHTS (Used for Chain) ===\033[0m")        
        
        raw_active = validator_instance.scores[validator_instance.scores > 1e-10]  # Use small threshold instead of 0
        if len(raw_active) > 0:
            bt.logging.info(f"Raw score range: {np.min(raw_active):.6f} - {np.max(raw_active):.6f}")
            raw_ratio = np.max(raw_active) / np.min(raw_active)
            bt.logging.info(f"Raw score ratio: {raw_ratio:.2f}")
            bt.logging.info(f"Active scores: {len(raw_active)} (filtered from {len(validator_instance.scores)})")
        else:
            bt.logging.warning("No significant raw scores found")
            return
        
        active_normalized = {}
        for uid, norm_score in enumerate(normalized_scores):
            if norm_score > 1e-6:  # Above your clipping threshold
                active_normalized[uid] = norm_score
        
        if not active_normalized:
            bt.logging.warning("No active normalized weights")
            return
            
        norm_array = np.array(list(active_normalized.values()))
        norm_stats = {
            'mean': np.mean(norm_array),
            'std': np.std(norm_array),
            'cv': np.std(norm_array) / np.mean(norm_array) if np.mean(norm_array) > 0 else 0,
            'min': np.min(norm_array),
            'max': np.max(norm_array),
            'sum': np.sum(norm_array)  # Should be ~1.0
        }
        
        bt.logging.info(f"Normalized sum: {norm_stats['sum']:.6f} (should be ~1.0)")
        bt.logging.info(f"Normalized CV: {norm_stats['cv']:.3f}")
        bt.logging.info(f"Normalized range: {norm_stats['min']:.6f} - {norm_stats['max']:.6f}")
        
        # Show top normalized weights
        sorted_normalized = sorted(active_normalized.items(), key=lambda x: x[1], reverse=True)
        bt.logging.info(f"Top 10 normalized weights:")
        for i, (uid, weight) in enumerate(sorted_normalized[:10], 1):
            percentage = weight * 100
            bt.logging.info(f"  {i}. UID {uid:2d}: {weight:.6f} ({percentage:.2f}%)")
        
        # Show which UIDs were filtered out
        zero_score_uids = [uid for uid, score in enumerate(validator_instance.scores) if score <= 1e-10]
        if zero_score_uids:
            bt.logging.info(f"Zero/near-zero score UIDs: {zero_score_uids}")
            
        # Check for weight concentration
        top_3_weight = sum(weight for _, weight in sorted_normalized[:3])
        if top_3_weight > 0.7:
            bt.logging.warning(f"⚠️  High weight concentration in top 3: {top_3_weight:.1%}")
        
        # Check for extreme ranges
        norm_ratio = norm_stats['max'] / norm_stats['min']
        if norm_ratio > 1000:
            bt.logging.warning(f"⚠️  Extreme normalized range - ratio: {norm_ratio:.2f}")
        
    except Exception as e:
        bt.logging.error(f"Error in normalized analysis: {e}")

def display_ema_insights(validator_instance):
    """Show EMA alpha usage patterns"""
    try:
        if not hasattr(validator_instance, 'alpha_history') or len(validator_instance.alpha_history) == 0:
            bt.logging.debug("No alpha history available")
            return
        
        recent_alphas = validator_instance.alpha_history[-10:]  # Last 10 updates
        avg_alpha = np.mean(recent_alphas)
        
        # Count usage patterns
        default_alpha = float(validator_instance.config.neuron.moving_average_alpha)
        #low_alpha_threshold = default_alpha * 0.6  # Assuming low_alpha is ~60% of default
        low_alpha_threshold = default_alpha / 2
        
        low_alpha_usage = sum(1 for a in recent_alphas if a < low_alpha_threshold) / len(recent_alphas)
        high_alpha_usage = sum(1 for a in recent_alphas if a > default_alpha) / len(recent_alphas)
        
        bt.logging.info(f"\033[1;36m=== EMA INSIGHTS ===\033[0m")
        bt.logging.info(f"Default alpha: {default_alpha:.3f}")
        bt.logging.info(f"Recent avg alpha: {avg_alpha:.3f}")
        bt.logging.info(f"Low alpha usage: {low_alpha_usage:.1%} (failure handling)")
        bt.logging.info(f"High alpha usage: {high_alpha_usage:.1%} (rapid updates)")
        
        # Show alpha trend
        if len(recent_alphas) >= 5:
            alpha_trend = np.polyfit(range(len(recent_alphas)), recent_alphas, 1)[0]
            bt.logging.info(f"Alpha trend: {alpha_trend:+.4f} {'(increasing)' if alpha_trend > 0 else '(decreasing)'}")
        
        # Interpret patterns
        if low_alpha_usage > 0.3:
            bt.logging.warning("⚠️  High failure rate detected (frequent low alpha usage)")
        if avg_alpha < default_alpha * 0.8:
            bt.logging.warning("⚠️  System in defensive mode (low average alpha)")
        
    except Exception as e:
        bt.logging.error(f"Error in EMA insights: {e}")

def display_transformation_impact(validator_instance):
    """Show impact of non-linear transformation"""
    try:
        if not hasattr(validator_instance, 'scores') or len(validator_instance.scores) == 0:
            return
        
        # Get active scores
        active_scores = validator_instance.scores[validator_instance.scores > 0]
        if len(active_scores) < 2:
            return
        
        # CRITICAL FIX: Filter out extremely small values before transformation
        min_threshold = 1e-6
        filtered_scores = active_scores[active_scores > min_threshold]
        
        if len(filtered_scores) < 2:
            bt.logging.warning("⚠️  Too few significant scores for transformation analysis")
            return
        
        # Normalize AFTER filtering
        normalized = filtered_scores / np.sum(filtered_scores)
        
        # Apply different powers
        linear = normalized  # Power = 1.0
        moderate = np.power(normalized, 1.2)
        aggressive = np.power(normalized, 1.5)
        
        # Renormalize
        moderate = moderate / np.sum(moderate)
        aggressive = aggressive / np.sum(aggressive)
        
        bt.logging.info(f"\033[1;36m=== TRANSFORMATION IMPACT ===\033[0m")
        bt.logging.info(f"Scores analyzed: {len(filtered_scores)} (filtered from {len(active_scores)})")
        bt.logging.info(f"Score range: {np.min(filtered_scores):.6f} - {np.max(filtered_scores):.6f}")
        
        # Safe ratio calculation
        linear_ratio = np.max(linear) / np.min(linear)
        moderate_ratio = np.max(moderate) / np.min(moderate)
        aggressive_ratio = np.max(aggressive) / np.min(aggressive)
        
        bt.logging.info(f"Linear (1.0) CV: {np.std(linear)/np.mean(linear):.3f}")
        bt.logging.info(f"Moderate (1.2) CV: {np.std(moderate)/np.mean(moderate):.3f}")
        bt.logging.info(f"Aggressive (1.5) CV: {np.std(aggressive)/np.mean(aggressive):.3f}")
        
        # Show current power being used
        current_power = 1.0  # Update this based on your actual setting
        bt.logging.info(f"Current power: {current_power} ({'linear' if current_power == 1.0 else 'non-linear'})")
        
        # Show amplification effect with safe values
        bt.logging.info(f"Max/Min ratios - Linear: {linear_ratio:.2f}, Moderate: {moderate_ratio:.2f}, Aggressive: {aggressive_ratio:.2f}")
        
        # Warning for extreme ratios
        if linear_ratio > 1000:
            bt.logging.warning(f"⚠️  Extreme score range detected - consider score capping")
        
    except Exception as e:
        bt.logging.error(f"Error in transformation impact: {e}")

def display_score_trends(validator_instance):
    """Display score trends over time"""
    try:
        if not hasattr(validator_instance, 'score_history') or len(validator_instance.score_history) < 2:
            return
            
        current = validator_instance.score_history[-1]
        previous = validator_instance.score_history[-2]
        
        # Calculate changes
        mean_change = current['stats']['mean'] - previous['stats']['mean']
        cv_change = current['stats']['cv'] - previous['stats']['cv']
        
        # Check for leadership changes
        current_leader = current['top_3'][0][0] if current['top_3'] else None
        previous_leader = previous['top_3'][0][0] if previous['top_3'] else None
        
        bt.logging.info(f"\033[1;35m=== SCORE TRENDS ===\033[0m")
        bt.logging.info(f"Mean change: {mean_change:+.6f}")
        bt.logging.info(f"CV change: {cv_change:+.4f} {'\033[32m(more stable)\033[0m' if cv_change < 0 else '\033[33m(less stable)\033[0m'}")
        
        if current_leader != previous_leader:
            bt.logging.info(f"🏆 Leadership change: UID {previous_leader} → UID {current_leader}")
        
        # Show UIDs that entered/left top 3
        current_top3_uids = {uid for uid, _ in current['top_3']}
        previous_top3_uids = {uid for uid, _ in previous['top_3']}
        
        new_top3 = current_top3_uids - previous_top3_uids
        dropped_top3 = previous_top3_uids - current_top3_uids
        
        if new_top3:
            bt.logging.info(f"📈 Entered top 3: {list(new_top3)}")
        if dropped_top3:
            bt.logging.info(f"📉 Dropped from top 3: {list(dropped_top3)}")
        
        # Show new/lost miners
        new_miners = set(current['active_uids']) - set(previous['active_uids'])
        lost_miners = set(previous['active_uids']) - set(current['active_uids'])
        
        if new_miners:
            bt.logging.info(f"🆕 New active miners: {list(new_miners)}")
        if lost_miners:
            bt.logging.info(f"❌ Lost miners: {list(lost_miners)}")
        
        # Calculate score stability over longer periods
        if len(validator_instance.score_history) >= 5:
            last_5_cvs = [snapshot['stats']['cv'] for snapshot in validator_instance.score_history[-5:]]
            cv_trend = np.polyfit(range(5), last_5_cvs, 1)[0]  # Linear trend
            bt.logging.info(f"CV trend (last 5): {cv_trend:+.4f} {'(stabilizing)' if cv_trend < 0 else '(destabilizing)'}")
    
    except Exception as e:
        bt.logging.error(f"Error in score trends: {e}")

def check_score_health(validator_instance, stats, max_min_ratio):
    """Enhanced health checks based on our convergence work"""
    try:
        health_issues = []
        recommendations = []
        
        # CV-based health (our main convergence metric)
        if stats['cv'] > 0.6:
            health_issues.append("Very high CV - poor convergence")
            recommendations.append("Consider increasing sample_size or alpha")
        elif stats['cv'] > 0.4:
            health_issues.append("High CV - moderate convergence issues")
            recommendations.append("Monitor for improvement, may need tuning")
        elif stats['cv'] < 0.3:
            bt.logging.info(f"✅ Excellent convergence (CV: {stats['cv']:.3f})")
        
        # Score divergence (our Atlantic validator issue)
        if max_min_ratio > 100:
            health_issues.append("Extreme score divergence - geographic/connectivity issues")
            recommendations.append("Check cross-validator consensus")
        elif max_min_ratio > 20:
            health_issues.append("High score divergence")
            recommendations.append("Consider score capping or investigate outliers")
        
        # Network size considerations
        if stats['count'] < 10:
            health_issues.append("Small network - testnet conditions")
            recommendations.append("May need different parameters than production")
        
        # Active miner stability
        if hasattr(validator_instance, 'score_history') and len(validator_instance.score_history) >= 3:
            recent_counts = [s['stats']['count'] for s in validator_instance.score_history[-3:]]
            if max(recent_counts) - min(recent_counts) > 5:
                health_issues.append("Unstable active miner count")
                recommendations.append("Check network connectivity")
        
        # EMA responsiveness check
        if hasattr(validator_instance, 'alpha_history') and len(validator_instance.alpha_history) >= 5:
            recent_alphas = validator_instance.alpha_history[-5:]
            avg_alpha = np.mean(recent_alphas)
            if avg_alpha < 0.3:
                health_issues.append("Low EMA responsiveness")
                recommendations.append("System may be in defensive mode due to failures")
        
        if health_issues:
            bt.logging.warning(f"🩺 Score Health Issues:")
            for issue in health_issues:
                bt.logging.warning(f"   - {issue}")
            
            if recommendations:
                bt.logging.info(f"💡 Recommendations:")
                for rec in recommendations:
                    bt.logging.info(f"   - {rec}")
        else:
            bt.logging.info(f"✅ Score health: Good")
    
    except Exception as e:
        bt.logging.error(f"Error in score health check: {e}")

def display_epoch_info(validator_instance):
    """Display epoch information for the current block"""
    try:
        current_block = validator_instance.block
        netuid = validator_instance.config.netuid
        
        current_epoch, blocks_until_next_epoch, epoch_start_block = epoch.get_current_epoch_info(current_block, netuid)

        block_time = 12
        minutes_to_next_block = blocks_until_next_epoch * block_time / 60
        
        bt.logging.info(f"\033[1;34m=== EPOCH INFO ===\033[0m")
        bt.logging.info(f"NetUID: {netuid}")
        bt.logging.info(f"Epoch start block: {epoch_start_block}")
        bt.logging.info(f"Current block: {current_block}")
        bt.logging.info(f"Current epoch: {current_epoch}")
        bt.logging.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        bt.logging.info(f"Minutes until next epoch: {minutes_to_next_block:.1f}")
        
    except Exception as e:
        bt.logging.error(f"Error in epoch info display: {e}")

def display_coverage_info(validator_instance):
    """Display coverage information for the current validator, accounting for unresponsive UIDs."""
    try:
        seen = getattr(validator_instance, "seen_uids", set())
        total = getattr(validator_instance, "total_uids", set())
        unresponsive = getattr(validator_instance, "unresponsive_uids", set())
        responsive = total - unresponsive if total else set()

        # Responsive coverage: how many responsive UIDs have been seen
        if responsive:
            responsive_coverage = len(seen & responsive) / len(responsive)
        else:
            responsive_coverage = 0.0

        # Total coverage: how many total UIDs have been seen
        total_coverage = len(seen) / len(total) if total else 0.0

        bt.logging.info(f"\033[1;35m=== COVERAGE INFO ===\033[0m")
        bt.logging.info(f"Seen UIDs (attempted): {len(seen)}")
        bt.logging.info(f"Total UIDs (metagraph): {len(total)}")
        bt.logging.info(f"Unresponsive UIDs: {len(unresponsive)}")
        bt.logging.info(f"Responsive UIDs: {len(responsive)}")
        bt.logging.info(f"Responsive coverage: {len(seen & responsive)}/{len(responsive)} ({responsive_coverage:.2%})")
        bt.logging.info(f"Total coverage (all time): {len(seen)}/{len(total)} ({total_coverage:.2%})")

        # Shades of green for responsive coverage
        if responsive_coverage < 0.10:
            bt.logging.warning(f"🟥 Very low responsive coverage: {responsive_coverage:.2%}")
        elif responsive_coverage < 0.25:
            bt.logging.warning(f"🟧 Low responsive coverage: {responsive_coverage:.2%}")
        elif responsive_coverage < 0.5:
            bt.logging.warning(f"🟨 Moderate responsive coverage: {responsive_coverage:.2%}")
        elif responsive_coverage < 0.7:
            bt.logging.info(f"🟩 Good responsive coverage: {responsive_coverage:.2%}")
        elif responsive_coverage < 0.85:
            bt.logging.info(f"🟩🟩 Very good responsive coverage: {responsive_coverage:.2%}")
        else:
            bt.logging.info(f"🟩🟩🟩 Excellent responsive coverage: {responsive_coverage:.2%}")

        bt.logging.info(f" -- Total Coverage Info -- ")
        # Shades of green for total coverage
        if total_coverage < 0.10:
            bt.logging.warning(f"🟥 Very low total coverage: {total_coverage:.2%}")
        elif total_coverage < 0.25:
            bt.logging.warning(f"🟧 Low total coverage: {total_coverage:.2%}")
        elif total_coverage < 0.5:
            bt.logging.warning(f"🟨 Moderate total coverage: {total_coverage:.2%}")
        elif total_coverage < 0.7:
            bt.logging.info(f"🟩 Good total coverage: {total_coverage:.2%}")
        elif total_coverage < 0.85:
            bt.logging.info(f"🟩🟩 Very good total coverage: {total_coverage:.2%}")
        else:
            bt.logging.info(f"🟩🟩🟩 Excellent total coverage: {total_coverage:.2%}")

    except Exception as e:
        bt.logging.error(f"Error in coverage info display: {e}")


def run_complete_score_analysis(validator_instance):
    """Run all score analysis functions in sequence"""
    try:
        bt.logging.info(f"\033[1;36m=== ENHANCED SCORE ANALYSIS ===\033[0m")
        
        # Run all analysis functions
        display_normalized_analysis(validator_instance)
        display_ema_insights(validator_instance)
        display_transformation_impact(validator_instance)
        display_score_trends(validator_instance)
        display_coverage_info(validator_instance)
        display_epoch_info(validator_instance)
        
        bt.logging.info(f"\033[1;36m=== ANALYSIS COMPLETE ===\033[0m")
        
    except Exception as e:
        bt.logging.error(f"Error in complete score analysis: {e}")
        bt.logging.error(traceback.format_exc())