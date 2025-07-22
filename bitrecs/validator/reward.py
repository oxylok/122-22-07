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
import hashlib
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

CONSENSUS_BONUS_MULTIPLIER = 1.01
SUSPECT_MINER_DECAY = 0.980
USE_DIFFICULTY_ADJUSTMENT = False


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


def verify_response_signature(response: BitrecsRequest) -> bool:
    try:
        if not response.miner_signature:
            bt.logging.error("Response missing miner_signature for verification.")
            return False
        payload = {
            "name": response.name,
            "axon_hotkey": response.axon.hotkey,
            "dendrite_hotkey": response.dendrite.hotkey,
            "created_at": response.created_at,
            "num_results": response.num_results,
            "query": response.query,
            "site_key": response.site_key,
            "results": response.results,
            "models_used": response.models_used,
            "miner_uid": response.miner_uid,
            "miner_hotkey": response.miner_hotkey,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).digest()
        signature = bytes.fromhex(response.miner_signature)
        miner_hotkey = response.miner_hotkey
        return bt.Keypair(ss58_address=miner_hotkey).verify(payload_hash, signature)
    except Exception as e:
        bt.logging.error(f"Error verifying response signature: {e}")
        return False


def calculate_miner_boost(hotkey: str, actions: List[UserAction]) -> float:
    """
    Reward miners who generate positive actions on ecommerce sites
    DISABLED DURING BOOTSTRAPPING PHASE

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
    validator_hotkey: str,
    ground_truth: BitrecsRequest,
    catalog_validator: CatalogValidator, 
    response: BitrecsRequest,
    actions: List[UserAction],
    r_limit: float = 1.0
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
        if response.dendrite.hotkey != validator_hotkey:
            bt.logging.error(f"Response from different hotkey: {response.dendrite.hotkey} != {validator_hotkey}")
            bt.logging.error(f"Miner hotkey: {response.axon.hotkey[:8]}")
            return 0.0
        if not response.dendrite.signature:
            bt.logging.error(f"Response missing signature: {response.axon.hotkey[:8]}")
            return 0.0
        if response.is_timeout:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_timeout is True, status: {response.dendrite.status_code}")
            return 0.0
        if response.is_failure:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_failure is True, status: {response.dendrite.status_code}")
            return 0.0
        if not response.is_success:
            bt.logging.error(f"{response.axon.hotkey[:8]} is_success is False, status: {response.dendrite.status_code}")
            return 0.0
        if not verify_response_signature(response):
            bt.logging.error(f"{response.axon.hotkey[:8]} response signature verification failed")
            return 0.0
        if not response.miner_uid or not response.miner_hotkey:
            bt.logging.error(f"{response.axon.hotkey[:8]} is not reporting correctly (missing ids)")
            return 0.0
        if response.miner_hotkey.lower() != response.axon.hotkey.lower():
            bt.logging.error(f"{response.miner_uid} hotkey mismatch: {response.miner_hotkey} != {response.axon.hotkey}")
            return 0.0
        if len(response.results) != ground_truth.num_results:
            bt.logging.error(f"{response.miner_uid} num_recs mismatch, expected {ground_truth.num_results} but got {len(response.results)}")
            return 0.0
        if len(response.models_used) != 1:
            bt.logging.error(f"{response.miner_uid} has invalid models used: {response.miner_hotkey[:8]}")
            return 0.0
        if response.axon.process_time < r_limit:
            bt.logging.error(f"\033[33m WARNING Miner {response.miner_uid} axon time: {response.axon.process_time} < {r_limit}\033[0m")
            return 0.0
        if response.dendrite.process_time < r_limit:
            bt.logging.error(f"\033[33m WARNING Miner {response.miner_uid} dendrite time: {response.dendrite.process_time} < {r_limit}\033[0m")
            return 0.0
        if response.query != ground_truth.query:
            bt.logging.error(f"{response.miner_uid} query mismatch: {response.query} != {ground_truth.query}")
            return 0.0
        if response.context != "[]":
            bt.logging.error(f"{response.miner_uid} context is not empty: {response.context}")
            return 0.0
        if not validate_result_schema(ground_truth.num_results, response.results):
            bt.logging.error(f"{response.miner_uid} failed schema validation: {response.miner_hotkey[:8]}")
            return 0.0
        
        valid_items = set()
        query_lower = response.query.lower().strip()
        for result in response.results:
            try:
                product = json_repair.loads(result)
                sku = product["sku"]
                if sku.lower() == query_lower:
                    bt.logging.error(f"{response.miner_uid} has query in results: {response.miner_hotkey[:8]}")
                    return 0.0
                if sku in valid_items:
                    bt.logging.error(f"{response.miner_uid} has duplicate results: {response.miner_hotkey[:8]}")
                    return 0.0
                if not catalog_validator.validate_sku(sku):
                    bt.logging.error(f"{response.miner_uid} has invalid results: {response.miner_hotkey[:8]}")
                    return 0.00
                
                valid_items.add(sku)
            except Exception as e:
                bt.logging.error(f"JSON ERROR: {e}, miner data: {response.miner_hotkey}")
                return 0.0

        if len(valid_items) != ground_truth.num_results:
            bt.logging.error(f"{response.miner_uid} invalid number of valid_items: {response.miner_hotkey[:8]}")
            return 0.0
        
        score = BASE_REWARD
        
        # if CONST.CONVERSION_SCORING_ENABLED and 1==2: #Disabled during boostrapping phase of mainnet
        #     # Adjust the rewards based on the actions
        #     boost = calculate_miner_boost(response.miner_hotkey, actions)
        #     if boost > 0:
        #         bt.logging.trace(f"\033[32m Miner {response.miner_uid} boost: {boost} \033[0m")
        #         bt.logging.trace(f"\033[32m current: {score} \033[0m")
        #         score = score + boost
        #         bt.logging.trace(f"\033[32m after: {score} \033[0m")
        #     else:
        #         bt.logging.trace(f"\033[33m Miner {response.miner_uid} boost: {boost} \033[0m")
        
        return score
    except Exception as e:
        bt.logging.error(f"Error in rewards: {e}, miner data: {response}")
        return 0.0


def get_rewards(
    validator_hotkey: str,
    ground_truth: BitrecsRequest,
    responses: List[BitrecsRequest],
    actions: List[UserAction] = None,
    r_limit: float = 1.0,
    batch_size: int = 16
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - ground_truth (:obj:`bitrecs.protocol.BitrecsRequest`): The original request object containing the query and context.
    - responses (List[:obj:`bitrecs.protocol.BitrecsRequest`]): The list of responses from miners.
    - actions (List[:obj:`bitrecs.commerce.user_action.UserAction`]): The list of user actions for the query.
    - r_limit (float): Min walltime for recs.
    - batch_size (int): Neuron sample size of batch.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    
    if ground_truth.num_results < CONST.MIN_RECS_PER_REQUEST or ground_truth.num_results > CONST.MAX_RECS_PER_REQUEST:
        bt.logging.error(f"Invalid number of recommendations: {ground_truth.num_results}")
        raise ValueError(f"Invalid number of recommendations: {ground_truth.num_results}")
        #return np.zeros(len(responses), dtype=float)
    
    store_catalog : list[Product] = ProductFactory.try_parse_context_strict(ground_truth.context)
    if len(store_catalog) < CONST.MIN_CATALOG_SIZE or len(store_catalog) > CONST.MAX_CATALOG_SIZE:
        bt.logging.error(f"Invalid catalog size: {len(store_catalog)}")
        raise ValueError(f"Invalid catalog size: {len(store_catalog)}")
        #return np.zeros(len(responses), dtype=float)
    catalog_validator = CatalogValidator(store_catalog)
    
    if not actions or len(actions) == 0:
        bt.logging.warning(f"\033[1;33m WARNING - no actions found in get_rewards \033[0m")
    
    axon_times = []
    for response in responses:
        axon_time = response.axon.process_time if response.axon and response.axon.process_time else None
        axon_times.append(axon_time)
    
    valid_times = [t for t in axon_times if t is not None and t > 0]    
    if len(valid_times) > 1:
        min_time = min(valid_times)
        max_time = max(valid_times)
        avg_time = sum(valid_times) / len(valid_times)
        spread = max_time - min_time
        bt.logging.trace(f"Batch: min={min_time:.4f}s, max={max_time:.4f}s, avg={avg_time:.4f}s, spread={spread:.4f}s")
        if spread > 2.0:            
            bt.logging.info(f"\033[33mWide Spread detected: {spread:.4f}s\033[0m")
    
    difficulty_decay = measure_request_difficulty(
        sku=ground_truth.query,
        catalog_size=len(store_catalog),
        num_recs=ground_truth.num_results,
        num_participants=len(responses),
        min_catalog_size=CONST.MIN_CATALOG_SIZE,
        max_catalog_size=CONST.MAX_CATALOG_SIZE,
        min_recs=CONST.MIN_RECS_PER_REQUEST,
        max_recs=CONST.MAX_RECS_PER_REQUEST,
        min_participants=1,
        max_participants=batch_size,
        base=1.0,
        min_decay=0.9,   # 10% penalty for easiest
        max_decay=1.0    # no penalty for hardest
    )

    difficulty_statement = get_difficulty_statement(difficulty_decay)
    bt.logging.trace(f"{difficulty_statement}")
    if not USE_DIFFICULTY_ADJUSTMENT:
        bt.logging.trace(f"\033[33mDifficulty adjustment skipped! \033[0m")

    rewards = []
    for i, response in enumerate(responses):        
        base_reward = reward(validator_hotkey, ground_truth, catalog_validator, response, actions, r_limit)        
        if base_reward <= 0.0:
            rewards.append(0.0)
            continue
        if USE_DIFFICULTY_ADJUSTMENT:
            final_score = base_reward * difficulty_decay
        else:
            final_score = base_reward
        rewards.append(final_score)
    
    return np.array(rewards, dtype=float)


def measure_request_difficulty(
    sku: str,
    catalog_size: int,
    num_recs: int,
    num_participants: int,
    min_catalog_size: int = 5,
    max_catalog_size: int = 1000,
    min_recs: int = CONST.MIN_RECS_PER_REQUEST,
    max_recs: int = CONST.MAX_RECS_PER_REQUEST,
    min_participants: int = 1,
    max_participants: int = 16,
    base: float = 1.0,
    min_decay: float = 0.9,   # 10% penalty for easiest
    max_decay: float = 1.0    # no penalty for hardest
) -> float:
    """
    Returns a decay factor in [min_decay, max_decay].
    - Easiest requests get min_decay (0.9).
    - Hardest requests get max_decay (1.0, no penalty).
    """
    catalog_weight = 0.4
    recs_weight = 0.1
    participants_weight = 0.2

    # Use catalog size instead of context length
    catalog_factor = (catalog_size - min_catalog_size) / (max_catalog_size - min_catalog_size)
    catalog_factor = max(0.0, min(catalog_factor, 1.0))
    recs_factor = (num_recs - min_recs) / (max_recs - min_recs)
    recs_factor = max(0.0, min(recs_factor, 1.0))
    part_factor = (num_participants - min_participants) / (max_participants - min_participants)
    part_factor = max(0.0, min(part_factor, 1.0))

    raw_difficulty = base * (1 + catalog_weight * catalog_factor) * (1 + recs_weight * recs_factor) * (1 + participants_weight * part_factor)
    max_difficulty = base * (1 + catalog_weight) * (1 + recs_weight) * (1 + participants_weight)

    # Map to decay factor in [min_decay, max_decay]
    decay = min_decay + (max_decay - min_decay) * (raw_difficulty - 1.0) / (max_difficulty - 1.0)
    decay = max(min_decay, min(decay, max_decay))
    return float(decay)


def get_difficulty_statement(difficulty: float) -> str:
    """
    Returns a human-readable statement about the request difficulty.
    """
    if difficulty <= 0.93:
        return f"Difficulty is very easy: \033[1;32m{difficulty:.3f}\033[0m"
    elif difficulty <= 0.97:
        return f"Difficulty is medium: \033[1;33m{difficulty:.3f}\033[0m"
    else:
        return f"Difficulty is hard: \033[1;31m{difficulty:.3f}\033[0m"