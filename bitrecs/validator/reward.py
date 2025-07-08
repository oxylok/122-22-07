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
import traceback
import numpy as np
import bittensor as bt
import jsonschema
import json_repair
from typing import List
from bitrecs.commerce.user_action import UserAction, ActionType
from bitrecs.protocol import BitrecsRequest
from bitrecs.commerce.product import Product, ProductFactory
from bitrecs.utils import constants as CONST

BASE_BOOST = 1/256
BASE_REWARD = 0.80
MAX_BOOST = 0.20
ALPHA_TIME_DECAY = 0.05
CONSENSUS_BONUS_MULTIPLIER = 1.025

ACTION_WEIGHTS = {
    ActionType.VIEW_PRODUCT.value: 0.05,
    ActionType.ADD_TO_CART.value: 0.10,
    ActionType.PURCHASE.value: 0.85,
}

class CatalogValidator:
    def __init__(self, store_catalog: List[Product]):
        self.sku_set = {product.sku.lower().strip() for product in store_catalog}
    
    def validate_sku(self, sku: str) -> bool:
        if not sku:
            return False
        return sku.lower().strip() in self.sku_set


def validate_result_schema(num_recs: int, results: list) -> bool:
    """
    Ensure results from Miner match the required schema
    """
    if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
        return False
    if len(results) != num_recs:
        bt.logging.error("Error validate_result_schema num_recs mismatch")
        return False
    
    schema = {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "price": {"type": ["string", "number"]},
            "reason": {"type": "string"}
        },
        "required": ["sku", "name", "price", "reason"],
    }

    count = 0
    for item in results:
        try:            
            #thing = json.loads(item)
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)           
            count += 1
        except json.decoder.JSONDecodeError as e:            
            bt.logging.trace(f"JSON JSONDecodeError ERROR: {e}")
            break
        except jsonschema.exceptions.ValidationError as e:            
            bt.logging.trace(f"JSON ValidationError ERROR: {e}")
            break
        except Exception as e:            
            bt.logging.trace(f"JSON Exception ERROR: {e}")
            break

    return count == len(results)


def calculate_miner_boost(hotkey: str, actions: List[UserAction]) -> float:
    """
    Reward miners who generate positive actions on ecommerce sites

    """
    try:
        if not actions or len(actions) == 0:
            return 0.0

        miner_actions = [a for a in actions if a["hot_key"].lower() == hotkey.lower()]
        if len(miner_actions) == 0:
            bt.logging.trace(f"Miner {hotkey} has no actions")
            return 0.0

        views = [v for v in miner_actions if v["action"] == ActionType.VIEW_PRODUCT.name]
        add_to_carts = [a for a in miner_actions if a["action"] == ActionType.ADD_TO_CART.name]
        purchases = [p for p in miner_actions if p["action"] == ActionType.PURCHASE.name]        

        if len(views) == 0 and len(add_to_carts) == 0 and len(purchases) == 0:
            bt.logging.trace(f"Miner {hotkey} has no parsed actions - skipping boost")
            return 0.0
        
        vf = ACTION_WEIGHTS[ActionType.VIEW_PRODUCT.value] * len(views)
        af = ACTION_WEIGHTS[ActionType.ADD_TO_CART.value] * len(add_to_carts)
        pf = ACTION_WEIGHTS[ActionType.PURCHASE.value] * len(purchases)
        total_boost = vf + af + pf
        bt.logging.trace(f"Miner {hotkey} total_boost: {total_boost} from views: ({len(views)}) add_to_carts: ({len(add_to_carts)}) purchases: ({len(purchases)})")

        # miner has no actions this round
        if total_boost == 0:
            return 0.0
        
        #TODO review this       
        if total_boost > BASE_BOOST:
            total_boost = MAX_BOOST / (1 + math.exp(-total_boost + BASE_BOOST))
        
        return min(max(total_boost, 0.0), MAX_BOOST)
    except Exception as e:
        bt.logging.error(f"Error in calculate_miner_boost: {e}")
        traceback.print_exc()
        return 0.0


def reward(
    num_recs: int, 
    catalog_validator: CatalogValidator, 
    response: BitrecsRequest,
    actions: List[UserAction]
) -> float:
    """
    Score the Miner's response to the BitrecsRequest 

    Nubmer of recommendations should match the requested number of recommendations
    Recommendations must exist in the original catalog
    Unique recommendations in the response is expected
    Malformed JSON or invliad skus will result in a 0.0 reward
    Miner rewards are boosted based on end-user actions on the ecommerce sites to encourage positive recs

    Returns:
    - float: The reward value for the miner.
    """
    
    try:
        score = 0.0
        if response.is_timeout:
            bt.logging.error(f"Miner {response.miner_uid} is_timeout is True, status: {response.dendrite.status_code}")
            return 0.0
        if response.is_failure:            
            bt.logging.error(f"Miner {response.miner_uid} is_failure is True, status: {response.dendrite.status_code}")
            return 0.0
        if not response.is_success:
            bt.logging.error(f"Miner {response.miner_uid} is_success is False, status: {response.dendrite.status_code}")
            return 0.0
        if len(response.results) != num_recs:
            bt.logging.error(f"Miner {response.miner_uid} num_recs mismatch, expected {num_recs} but got {len(response.results)}")
            return 0.0
        if not validate_result_schema(num_recs, response.results):
            bt.logging.error(f"Miner {response.miner_uid} failed schema validation: {response.miner_hotkey}")
            return 0.0
        if len(response.models_used) != 1:
            bt.logging.error(f"Miner {response.miner_uid} has multiple models used: {response.miner_hotkey}")
            return 0.0        
        if not CONST.RE_MODEL.match(response.models_used[0]):
            #miner_model = CONST.RE_REASON.sub("", response.models_used[0])
            bt.logging.error(f"Miner {response.miner_uid} has invalid model used: {response.miner_hotkey}, model: {response.models_used[0]}")
            return 0.0
        
        valid_items = set()
        query_lower = response.query.lower().strip()
        for result in response.results:
            try:
                product = json_repair.loads(result)
                sku = product["sku"]
                if sku.lower() == query_lower:
                    bt.logging.warning(f"Miner {response.miner_uid} has query in results: {response.miner_hotkey}")
                    return 0.0
                if sku in valid_items:
                    bt.logging.warning(f"Miner {response.miner_uid} has duplicate results: {response.miner_hotkey}")
                    return 0.0
                if not catalog_validator.validate_sku(sku):
                    bt.logging.warning(f"Miner {response.miner_uid} has invalid results: {response.miner_hotkey}")
                    return 0.00
                
                valid_items.add(sku)
            except Exception as e:
                bt.logging.error(f"JSON ERROR: {e}, miner data: {response.miner_hotkey}")
                return 0.0

        if len(valid_items) != num_recs:
            bt.logging.warning(f"Miner {response.miner_uid} invalid number of valid_items: {response.miner_hotkey}")
            return 0.0

        score = BASE_REWARD

        # For now just warn suspect miners
        plimit = 1.0
        if response.axon.process_time < plimit or response.dendrite.process_time < plimit:
            bt.logging.warning(f"\033[33m WARNING Miner {response.miner_uid} suspect time: {response.axon.process_time} \033[0m")
        
        if CONST.CONVERSION_SCORING_ENABLED and 1==2: #Disabled during boostrapping phase of mainnet
            # Adjust the rewards based on the actions
            boost = calculate_miner_boost(response.miner_hotkey, actions)
            if boost > 0:
                bt.logging.trace(f"\033[32m Miner {response.miner_uid} boost: {boost} \033[0m")
                bt.logging.trace(f"\033[32m current: {score} \033[0m")
                score = score + boost
                bt.logging.trace(f"\033[32m after: {score} \033[0m")
            else:
                bt.logging.trace(f"\033[33m Miner {response.miner_uid} boost: {boost} \033[0m")
        
        return score
    except Exception as e:
        bt.logging.error(f"Error in rewards: {e}, miner data: {response}")
        return 0.0


def get_rewards(
    num_recs: int,
    ground_truth: BitrecsRequest,
    responses: List[BitrecsRequest],
    actions: List[UserAction] = None
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - num_recs (int): The number of results expected per miner response.
    - ground_truth (BitrecsRequest): The original ground truth which contains the catalog and query
    - responses (List[float]): A list of responses from the miners.
    - actions (List[UserAction]): A list of user actions across all miners. 

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """

    if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
        bt.logging.error(f"Invalid number of recommendations: {num_recs}")
        return np.zeros(len(responses), dtype=float)
    
    store_catalog : list[Product] = ProductFactory.try_parse_context_strict(ground_truth.context)
    if len(store_catalog) < CONST.MIN_CATALOG_SIZE or len(store_catalog) > CONST.MAX_CATALOG_SIZE:
        bt.logging.error(f"Invalid catalog size: {len(store_catalog)}")
        return np.zeros(len(responses), dtype=float)
    catalog_validator = CatalogValidator(store_catalog)
    
    if not actions or len(actions) == 0:
        bt.logging.warning(f"\033[1;33m WARNING - no actions found in get_rewards \033[0m")

    # Extract all axon times first for batch processing
    axon_times = []
    for response in responses:
        axon_time = response.axon.process_time if response.axon and response.axon.process_time else None
        axon_times.append(axon_time)     
    
    # Filter out None values for percentile calculation
    valid_times = [t for t in axon_times if t is not None]

    # Log batch timing statistics
    if len(valid_times) > 1:
        min_time = min(valid_times)
        max_time = max(valid_times)
        avg_time = sum(valid_times) / len(valid_times)
        spread = max_time - min_time
        bt.logging.trace(f"Batch timing: min={min_time:.3f}s, max={max_time:.3f}s, avg={avg_time:.3f}s, spread={spread:.3f}s")
        
        # Log penalty impact
        if spread > 1.0:
            bt.logging.info(f"\033[33m High timing spread detected: {spread:.3f}s - penalties will be more significant \033[0m")
    
    # Calculate base rewards and apply batch-normalized timing penalties
    rewards = []
    for i, response in enumerate(responses):
        # Get base reward without timing penalty
        base_reward = reward(num_recs, catalog_validator, response, actions)
        
        if base_reward <= 0.0:
            rewards.append(0.0)
            continue
        miner_id = response.miner_uid if response.miner_uid is not None else response.miner_hotkey
        timing_penalty = 0.0
        # Apply percentile-based timing penalty
        if axon_times[i] is not None and len(valid_times) > 1:
            timing_penalty = calculate_percentile_timing_penalty(
                axon_times[i], valid_times, miner_id
            )
            final_reward = base_reward - timing_penalty
            rewards.append(max(final_reward, 0.0))  # Ensure non-negative
        elif axon_times[i] is None:
            # Penalty for missing timing data
            bt.logging.error(f"No axon_time found for miner {response.miner_uid} - hotkey {response.miner_hotkey}")
            timing_penalty = base_reward * 0.5  # Calculate the actual penalty amount
            rewards.append(base_reward * 0.5)
        else:
            # Only one valid response, no relative comparison possible
            bt.logging.trace(f"Single response batch - no timing penalty for miner {response.miner_uid}")
            rewards.append(base_reward)

        bt.logging.trace(f"UID {response.miner_uid} axon decay: {timing_penalty:.4f}, final_reward: {rewards[-1]:.4f}")

    return np.array(rewards, dtype=float)
   

def calculate_percentile_timing_penalty(axon_time: float, all_times: list, miner_uid: str) -> float:    
    if len(all_times) < 2:
        return ALPHA_TIME_DECAY * 0.5
    
    sorted_times = sorted(all_times)
    count_below = sum(1 for t in sorted_times if t < axon_time)
    count_equal = sum(1 for t in sorted_times if t == axon_time)
    rank = count_below + (count_equal + 1) / 2
    percentile = rank / len(all_times)
    
    # More forgiving curve - only penalize the slowest miners significantly
    if percentile <= 0.5:
        # Top 50% get minimal penalty
        penalty = ALPHA_TIME_DECAY * 0.1 * percentile
    else:
        # Bottom 50% get increasing penalty
        penalty = ALPHA_TIME_DECAY * (0.05 + 0.95 * (percentile - 0.5) * 2)
    
    bt.logging.trace(f"Miner {miner_uid} timing: {axon_time:.3f}s, percentile: {percentile:.2f}, penalty: {penalty:.4f}")
    return penalty
    