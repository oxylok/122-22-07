# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Bitrecs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import json
import numpy as np
import bittensor as bt
import json_repair
from typing import List, Dict, Set, Tuple
from bitrecs.commerce.user_action import UserAction, ActionType
from bitrecs.protocol import BitrecsRequest
from bitrecs.commerce.product import Product, ProductFactory
from bitrecs.utils import constants as CONST

BASE_BOOST = 1/256
BASE_REWARD = 0.80
MAX_BOOST = 0.20
ALPHA_TIME_DECAY = 0.05

ACTION_WEIGHTS = {
    ActionType.VIEW_PRODUCT.value: 0.05,
    ActionType.ADD_TO_CART.value: 0.10,
    ActionType.PURCHASE.value: 0.85,
}

# Pre-compiled schema for validation
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string"},
        "name": {"type": "string"},
        "price": {"type": ["string", "number"]},
        "reason": {"type": "string"}
    },
    "required": ["sku", "name", "price", "reason"],
}

class CatalogValidator:
    def __init__(self, store_catalog: List[Product]):        
        self.sku_set = {product.sku.lower().strip() for product in store_catalog}
        self.catalog_size = len(store_catalog)
    
    def validate_sku(self, sku: str) -> bool:        
        return bool(sku) and sku.lower().strip() in self.sku_set
    
    def validate_skus_batch(self, skus: List[str]) -> List[bool]:        
        if not skus:
            return []        
        
        normalized_skus = [sku.lower().strip() if sku else "" for sku in skus]
        return [bool(sku) and sku in self.sku_set for sku in normalized_skus]


class ActionBoostCalculator:
    """Pre-compute and cache action boosts for miners"""
    
    def __init__(self, actions: List[UserAction]):
        self.miner_actions_cache = self._group_actions_by_miner(actions)
    
    def _group_actions_by_miner(self, actions: List[UserAction]) -> Dict[str, Dict[str, int]]:        
        if not actions:
            return {}
        
        miner_stats = {}
        for action in actions:
            hotkey = action.get("hot_key", "").lower()
            if not hotkey:
                continue
                
            if hotkey not in miner_stats:
                miner_stats[hotkey] = {
                    ActionType.VIEW_PRODUCT.name: 0,
                    ActionType.ADD_TO_CART.name: 0,
                    ActionType.PURCHASE.name: 0
                }
            
            action_type = action.get("action")
            if action_type in miner_stats[hotkey]:
                miner_stats[hotkey][action_type] += 1
        
        return miner_stats
    
    def get_boost(self, hotkey: str) -> float:
        """Get cached boost for miner"""
        hotkey_lower = hotkey.lower()
        if hotkey_lower not in self.miner_actions_cache:
            return 0.0
        
        stats = self.miner_actions_cache[hotkey_lower]        
        
        views = stats[ActionType.VIEW_PRODUCT.name]
        add_to_carts = stats[ActionType.ADD_TO_CART.name] 
        purchases = stats[ActionType.PURCHASE.name]
        
        if views == 0 and add_to_carts == 0 and purchases == 0:
            return 0.0
        
        total_boost = (
            ACTION_WEIGHTS[ActionType.VIEW_PRODUCT.value] * views +
            ACTION_WEIGHTS[ActionType.ADD_TO_CART.value] * add_to_carts +
            ACTION_WEIGHTS[ActionType.PURCHASE.value] * purchases
        )
        
        if total_boost == 0:
            return 0.0
        
        if total_boost > BASE_BOOST:
            total_boost = MAX_BOOST / (1 + math.exp(-total_boost + BASE_BOOST))
        
        return min(max(total_boost, 0.0), MAX_BOOST)


def validate_result_schema_fast(num_recs: int, results: list) -> Tuple[bool, int]:
    """
    Fast schema validation with early returns and minimal exception handling
    Returns (is_valid, valid_count)
    """    
    if not (1 <= num_recs <= CONST.MAX_RECS_PER_REQUEST):
        return False, 0
    
    if len(results) != num_recs:
        bt.logging.error("Error validate_result_schema num_recs mismatch")
        return False, 0
    
    valid_count = 0
    
    for item in results:
        try:            
            try:
                parsed_item = json.loads(item)
            except json.JSONDecodeError:
                parsed_item = json_repair.loads(item)            
            
            if not all(key in parsed_item for key in ["sku", "name", "price", "reason"]):
                break            
            
            if (not isinstance(parsed_item["sku"], str) or 
                not isinstance(parsed_item["name"], str) or
                not isinstance(parsed_item["reason"], str) or
                not isinstance(parsed_item["price"], (str, int, float))):
                break
            
            valid_count += 1
            
        except Exception as e:
            bt.logging.trace(f"JSON validation error: {e}")
            break
    
    return valid_count == len(results), valid_count


def validate_response_fast(
    response: BitrecsRequest, 
    num_recs: int, 
    catalog_validator: CatalogValidator,
    query_lower: str
) -> Tuple[bool, str, Set[str]]:
    """
    Fast response validation with batched operations
    Returns (is_valid, error_message, valid_skus)
    """
    
    if response.is_timeout:
        return False, f"Miner {response.miner_uid} timeout", set()
    
    if response.is_failure:
        return False, f"Miner {response.miner_uid} failure", set()
    
    if not response.is_success:
        return False, f"Miner {response.miner_uid} not successful", set()
    
    if len(response.results) != num_recs:
        return False, f"Miner {response.miner_uid} num_recs mismatch", set()
    
    # Schema validation
    is_valid_schema, _ = validate_result_schema_fast(num_recs, response.results)
    if not is_valid_schema:
        return False, f"Miner {response.miner_uid} failed schema validation", set()
    
    # Parse all results and extract SKUs
    skus = []
    try:
        for result in response.results:
            try:
                product = json.loads(result)
            except json.JSONDecodeError:
                product = json_repair.loads(result)
            skus.append(product["sku"])
    except Exception as e:
        return False, f"JSON parsing error: {e}", set()
    
    # Batch validation checks
    if not skus:
        return False, "No SKUs found", set()
    
    # Check for query in results (batch)
    skus_lower = [sku.lower().strip() for sku in skus]
    if query_lower in skus_lower:
        return False, f"Miner {response.miner_uid} has query in results", set()
    
    # Check for duplicates
    unique_skus = set(skus)
    if len(unique_skus) != len(skus):
        return False, f"Miner {response.miner_uid} has duplicate results", set()
    
    # Batch SKU validation
    sku_validations = catalog_validator.validate_skus_batch(skus)
    if not all(sku_validations):
        return False, f"Miner {response.miner_uid} has invalid SKUs", set()
    
    return True, "", unique_skus


def reward_fast(
    num_recs: int, 
    catalog_validator: CatalogValidator, 
    response: BitrecsRequest,
    boost_calculator: ActionBoostCalculator,
    query_lower: str
) -> float:
    """
    Optimized reward calculation with minimal logging and batched operations
    """
    try:
        
        is_valid, error_msg, valid_skus = validate_response_fast(
            response, num_recs, catalog_validator, query_lower
        )
        
        if not is_valid:
            if error_msg:
                bt.logging.error(error_msg)
            return 0.0
        
        if len(valid_skus) != num_recs:
            bt.logging.warning(f"Miner {response.miner_uid} invalid number of valid_items")
            return 0.0
        
        # Base score calculation
        score = BASE_REWARD
        
        # Time penalty
        headers = response.to_headers()
        dendrite_time_header = headers.get("bt_header_dendrite_process_time")
        
        if dendrite_time_header:
            dendrite_time = float(dendrite_time_header)
            bt.logging.trace(f"Miner {response.miner_uid} dendrite_time: {dendrite_time}")
            
            if dendrite_time < 1.0:
                bt.logging.trace(f"WARNING Miner {response.miner_uid} suspect dendrite_time: {dendrite_time}")
            
            score = score - ALPHA_TIME_DECAY * dendrite_time
        else:
            bt.logging.error("Error in reward: dendrite_time not found in headers")
            return 0.0
        
        # Action boost (only if enabled)
        if CONST.CONVERSION_SCORING_ENABLED:
            boost = boost_calculator.get_boost(response.miner_hotkey)
            if boost > 0:
                bt.logging.trace(f"Miner {response.miner_uid} boost: {boost}")
                score += boost
        
        bt.logging.info(f"Final score: {score}")
        return score
        
    except Exception as e:
        bt.logging.error(f"Error in reward_fast: {e}")
        return 0.0


def get_rewards_optimized(
    num_recs: int,
    ground_truth: BitrecsRequest,
    responses: List[BitrecsRequest],
    actions: List[UserAction] = None
) -> np.ndarray:
    """
    Optimized version with pre-computation and batched operations
    """
    # Early validation
    if not (1 <= num_recs <= CONST.MAX_RECS_PER_REQUEST):
        bt.logging.error(f"Invalid number of recommendations: {num_recs}")
        return np.zeros(len(responses), dtype=float)
    
    # Parse catalog once
    try:
        store_catalog = ProductFactory.try_parse_context_strict(ground_truth.context)
    except Exception as e:
        bt.logging.error(f"Failed to parse catalog: {e}")
        return np.zeros(len(responses), dtype=float)
    
    catalog_size = len(store_catalog)
    if not (CONST.MIN_CATALOG_SIZE <= catalog_size <= CONST.MAX_CATALOG_SIZE):
        bt.logging.error(f"Invalid catalog size: {catalog_size}")
        return np.zeros(len(responses), dtype=float)
    
    # Pre-compute validation objects
    catalog_validator = CatalogValidator(store_catalog)
    boost_calculator = ActionBoostCalculator(actions or [])
    query_lower = ground_truth.query.lower().strip()
    
    if not actions:
        bt.logging.warning("WARNING - no actions found in get_rewards")
    
    # Batch process all responses
    rewards = np.zeros(len(responses), dtype=float)
    
    for i, response in enumerate(responses):
        rewards[i] = reward_fast(
            num_recs, 
            catalog_validator, 
            response,
            boost_calculator,
            query_lower
        )
    
    return rewards


# Backward compatibility
def calculate_miner_boost(hotkey: str, actions: List[UserAction]) -> float:
    """Backward compatibility wrapper"""
    boost_calculator = ActionBoostCalculator(actions or [])
    return boost_calculator.get_boost(hotkey)


def validate_result_schema(num_recs: int, results: list) -> bool:
    """Backward compatibility wrapper"""
    is_valid, _ = validate_result_schema_fast(num_recs, results)
    return is_valid


def reward(
    num_recs: int, 
    catalog_validator: CatalogValidator, 
    response: BitrecsRequest,
    actions: List[UserAction]
) -> float:
    """Backward compatibility wrapper"""
    boost_calculator = ActionBoostCalculator(actions or [])
    query_lower = response.query.lower().strip()
    return reward_fast(num_recs, catalog_validator, response, boost_calculator, query_lower)


def get_rewards(
    num_recs: int,
    ground_truth: BitrecsRequest,
    responses: List[BitrecsRequest],
    actions: List[UserAction] = None
) -> np.ndarray:
    """Use optimized version by default"""
    return get_rewards_optimized(num_recs, ground_truth, responses, actions)


