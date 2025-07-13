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

import time
import bittensor as bt
import random
from datetime import datetime, timezone
from bitrecs.protocol import BitrecsRequest
from bitrecs.validator.reward import get_rewards
from bitrecs.utils.uids import get_random_miner_uids


def get_bitrecs_dummy_request(num_results) -> BitrecsRequest:
    """
    Returns a dummy BitrecsRequest object for testing purposes.

    """
  
    queries = ["WT02-M-Green", "WT05-L-Purple", "24-MB02", "WSH11-28-Blue", "MH11"]
    query = random.choice(queries)

    json_context = "[]"
    utc_now = datetime.now(timezone.utc)
    created_at = utc_now.strftime("%Y-%m-%dT%H:%M:%S")

    p = BitrecsRequest(user="user1", 
                       query=query, 
                       context=json_context, 
                       created_at=created_at, 
                       num_results=num_results, 
                       site_key="site1", 
                       results=[""], 
                       models_used=[""], 
                       miner_hotkey="", 
                       miner_uid="")
    return p


async def forward(self, pr: BitrecsRequest = None):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        pr (:obj:`bitrecs.protocol.BitrecsRequest`): The end user request object to be sent to the network (from API)

    """
    bt.logging.info(f"VALIDATOR FORWARD Forwarding request: {pr}")
    
    num_results = 0
    if pr is not None: #API REQUEST
        raise NotImplementedError("API requests not implemented here")
     
    else:
        num_results = 5
        next_request = get_bitrecs_dummy_request(num_results)

    num_recs = next_request.num_results
    miner_uids = get_random_miner_uids(self,  k=self.sample_size)    

    bt.logging.info(f"** UID uids: {miner_uids}")
    start_time = time.time()

    # The dendrite client queries the network.
    responses = await self.dendrite(        
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=next_request,
        deserialize=False,
    )
    end_time = time.time()
    wall_time = end_time - start_time
    bt.logging.info(f"forward Wall time: {wall_time}")
    
    bt.logging.info(f"Received {len(responses)} responses: {responses}")
    
    # Adjust the scores based on responses from miners.
    rewards = get_rewards(num_recs=num_recs, 
                          ground_truth=next_request,
                           responses=responses)    
    
    if not len(miner_uids) == len(responses) == len(rewards):
        bt.logging.error(f"MISMATCH Error in rewards: {rewards}, responses: {responses}, miner_uids: {miner_uids}")

    bt.logging.info(f"Scored responses: {rewards}")
    
    #TODO synthetics disabled for now
    #self.update_scores(rewards, miner_uids)