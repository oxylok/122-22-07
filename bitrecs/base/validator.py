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

import os
import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
import time
import traceback
import anyio.to_thread
import wandb
import anyio
import subprocess
from dotenv import load_dotenv
load_dotenv()

from random import SystemRandom
safe_random = SystemRandom()
from typing import List, Union, Optional
from dataclasses import dataclass
from queue import SimpleQueue, Empty
from bitrecs.base.neuron import BaseNeuron
from bitrecs.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit, 
)
from bitrecs.utils import constants as CONST
from bitrecs.utils.config import add_validator_args
from bitrecs.api.api_server import ApiServer
from bitrecs.protocol import BitrecsRequest
from bitrecs.utils.distance import (
    display_rec_matrix_numpy,
    rec_list_to_set, 
    select_most_similar_bitrecs
)
from bitrecs.validator.reward import CONSENSUS_BONUS_MULTIPLIER, get_rewards
from bitrecs.validator.rules import validate_br_request
from bitrecs.utils.logging import (    
    log_miner_responses_to_sql,
    write_node_info
)
from bitrecs.utils.wandb import WandbHelper
from bitrecs.commerce.user_action import UserAction


API_QUEUE = SimpleQueue() # Queue of SynapseEventPair

@dataclass
class SynapseWithEvent:
    """ Object that API server can send to main thread to be serviced. """
    input_synapse: BitrecsRequest
    event: threading.Event
    output_synapse: BitrecsRequest


async def api_forward(synapse: BitrecsRequest) -> BitrecsRequest:
    """ Forward function for API server. """
    bt.logging.trace(f"API FORWARD validator synapse type: {type(synapse)}")
    synapse_with_event = SynapseWithEvent(
        input_synapse=synapse,
        event=threading.Event(),
        output_synapse=BitrecsRequest(
            name=synapse.name,                     
            created_at=synapse.created_at,
            user="",
            num_results=synapse.num_results,
            query=synapse.query,
            context="",
            site_key="",
            results=[""],
            models_used=[""],
            miner_uid="",
            miner_hotkey=""
        )
    )
    API_QUEUE.put(synapse_with_event)
    # Wait until the main thread marks this synapse as processed.
    await anyio.to_thread.run_sync(synapse_with_event.event.wait)
    return synapse_with_event.output_synapse


class BaseValidatorNeuron(BaseNeuron):
    """
    Validator for Bitrecs
    """

    neuron_type: str = "ValidatorNeuron"
    
    first_sync = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)
        np.set_printoptions(suppress=True, precision=8, linewidth=80)
        
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)      

        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")
            raise Exception("Axon off, not serving ip to chain.")
        
        api_port = int(os.environ.get("VALIDATOR_API_PORT"))
        if api_port != 7779:
            raise Exception("API Port must be set to 7779")
        
        self.api_port = api_port
        self.api_server = None
        if self.config.api.enabled:            
            self.api_server = ApiServer(
                api_port=self.api_port,
                forward_fn=api_forward,
                validator=self
            )
            self.api_server.start()
            bt.logging.info(f"\033[1;32m 🐸 API Endpoint Started: http://{self.api_server.config.host}:{self.api_server.config.port} \033[0m")
        else:            
            bt.logging.error(f"\033[1;31m No API Endpoint \033[0m")
        
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        self.active_miners: List[int] = []
        self.network = os.environ.get("NETWORK").strip().lower() #localnet / testnet / mainnet        
        self.user_actions: List[UserAction] = []
        
        write_node_info(
            network=self.network,
            uid=self.uid,
            hotkey=self.wallet.hotkey.ss58_address,
            neuron_type=self.neuron_type,
            sample_size=self.config.neuron.sample_size,
            v_limit=self.config.neuron.vpermit_tao_limit
        )

        if self.config.wandb.enabled == True:
            wandb_project = f"bitrecs_{self.network}"
            wandb_entity = self.config.wandb.entity
            if len(wandb_project) == 0 or len(wandb_entity) == 0:
                bt.logging.error("Wandb project name not set")
                raise Exception("Wandb project name not set")
            else:
                wandb_config = {
                    "network": self.network,
                    "neuron_type": self.neuron_type,
                    "sample_size": self.config.neuron.sample_size,
                    "num_concurrent_forwards": 1,
                    "vpermit_tao_limit": self.config.neuron.vpermit_tao_limit,
                    "run_name": f"validator_{wandb.util.generate_id()}"
                }
                self.wandb = WandbHelper(
                    project_name=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config
                )

        bt.logging.info(f"Validator Initialized at block: {self.block}")


    def serve_axon(self):
        """Serve axon to enable external connections."""
        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config, port=self.config.axon.port)
            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass


    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)


    async def analyze_similar_requests(self, num_recs: int, requests: List[BitrecsRequest]) -> Optional[List[BitrecsRequest]]:
        if not requests or len(requests) < 2:
            bt.logging.warning(f"Too few requests to analyze: {len(requests)} on step {self.step}")
            return
        
        async def get_dynamic_top_n(num_requests: int) -> int:        
            if num_requests < 4:
                return 2                     
            suggested = max(2, min(5, num_requests // 3))
            return suggested
        
        if self.config.logging.trace:
            print(f"Starting analyze_similar_requests with step: {self.step} and num_recs: {num_recs}")
        
        st = time.perf_counter()
        try:
            requests = [r for r in requests if r.is_success]
            valid_requests = []
            valid_recs = []
            models_used = []
            for br in requests:
                try:
                    headers = br.to_headers()
                    dendrite_time = 0
                    if "bt_header_dendrite_process_time" in headers:
                        dendrite_time = float(headers["bt_header_dendrite_process_time"])
                    skus = rec_list_to_set(br.results)
                    if skus:
                        valid_requests.append(br)
                        valid_recs.append(skus)
                        this_model = br.models_used[0] if br.models_used else "unknown"
                        if dendrite_time < 1:
                            this_model = f"{this_model} - X"
                        models_used.append(this_model)
                        
                except Exception as e:
                    bt.logging.error(f"Error extracting SKUs from results: {e}")
                    continue                    
            if not valid_recs:
                bt.logging.error(f"\033[1;33m No valid recs found to analyze on step: {self.step} \033[0m")
                return
            
            top_n = await get_dynamic_top_n(len(valid_requests))
            bt.logging.info(f"\033[1;32m Top {top_n} of {len(valid_requests)}/{len(requests)} (valid/total) bitrecs \033[0m")
            most_similar = select_most_similar_bitrecs(valid_requests, top_n)
            if not most_similar:
                bt.logging.warning(f"\033[33m No similar recs found in this round step: {self.step} \033[0m")
                return
            for sim in most_similar:
                bt.logging.info(f"\033[32m Miner {sim.miner_uid} {sim.models_used}\033[0m - batch: {sim.site_key}")

            if self.config.logging.trace:
                most_similar_indices = [valid_requests.index(req) for req in most_similar]
                matrix = display_rec_matrix_numpy(valid_recs, models_used, highlight_indices=most_similar_indices)            
                bt.logging.trace(matrix)

            et = time.perf_counter()
            diff = et - st
            bt.logging.info(f"Time taken to analyze similar bitrecs: \033[33m{diff:.2f}\033[0m seconds")
            return most_similar
        
        except Exception as e:            
            bt.logging.error(f"analyze_similar_requests failed with exception: {e}")            
            bt.logging.error(traceback.format_exc())
            return
        
        
    def cleanup_connections(self, axons=None):
        """Force cleanup of stale connections to specific miner ports"""
        try:
            ports_to_clean = set()
            if not axons:
                return
            
            for axon in axons:
                try:
                    port = int(axon.port)
                    ports_to_clean.add(port)
                except (ValueError, AttributeError):
                    continue
            
            if not ports_to_clean:
                bt.logging.warning("No valid miner ports found for cleanup")
                return
                
            bt.logging.trace(f"Cleaning up connections to ports: {sorted(ports_to_clean)}")
            for port in ports_to_clean:
                subprocess.run(['ss', '-K', 'state', 'close-wait', 'dport', '=', f':{port}'], 
                            capture_output=True, check=False)
                subprocess.run(['ss', '-K', 'state', 'last-ack', 'dport', '=', f':{port}'], 
                            capture_output=True, check=False)
            
            bt.logging.trace(f"Cleaned up stale connections to {len(ports_to_clean)} miner ports")            
        except Exception as e:
            bt.logging.error(f"Failed to cleanup connections: {e}")
            pass
        

    async def main_loop(self):
        """Main loop for the validator."""
        bt.logging.info(
            f"\033[1;32m 🐸 Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}\033[0m")
        if hasattr(self, "axon"):
            f"Axon: {self.axon}"
        
        bt.logging.info(f"Validator starting at block: {self.block}")
        bt.logging.info(f"Validator SAMPLE SIZE: {self.config.neuron.sample_size}")
        try:
            while True:
                try:

                    api_enabled = self.config.api.enabled
                    api_exclusive = self.config.api.exclusive
                    bt.logging.trace(f"api_enabled: {api_enabled} | api_exclusive {api_exclusive}")

                    synapse_with_event: Optional[SynapseWithEvent] = None
                    try:
                        synapse_with_event = API_QUEUE.get()
                        bt.logging.info(f"NEW API REQUEST {synapse_with_event.input_synapse.name}")
                    except Empty:
                        # No synapse from API server.
                        pass #continue prevents regular val loop

                    if synapse_with_event is not None and api_enabled: #API request
                        bt.logging.info("** Processing synapse from API server **")                        
                        bt.logging.info(f"Queue Size: {API_QUEUE.qsize()}")

                        # Validate the input synapse
                        if not validate_br_request(synapse_with_event.input_synapse):
                            bt.logging.error("Request failed Validation, skipped.")
                            synapse_with_event.event.set()
                            continue
                        
                        chosen_uids : list[int] = self.active_miners                     
                        if len(chosen_uids) == 0:
                            bt.logging.error("\033[31m API Request- No active miners, skipping - check your connectivity \033[0m")
                            synapse_with_event.event.set()
                            continue
                        bt.logging.trace(f"chosen_uids: {chosen_uids}")

                        chosen_axons = [self.metagraph.axons[uid] for uid in chosen_uids]
                        api_request = synapse_with_event.input_synapse
                        number_of_recs_desired = api_request.num_results
                        
                        st = time.perf_counter()
                        responses = await self.dendrite.forward(
                            axons = chosen_axons, 
                            synapse = api_request,
                            timeout = min(5, CONST.MAX_DENDRITE_TIMEOUT),
                            deserialize=False,
                            run_async=True
                        )
                        #TODO: 503 error handling async bug?
                        any_success = any([r for r in responses if r.is_success])
                        if not any_success:
                            bt.logging.error("\033[1;33mRETRY ATTEMPT\033[0m")
                            self.cleanup_connections(axons=chosen_axons)
                            responses = await self.dendrite.forward(
                                axons = chosen_axons, 
                                synapse = api_request,
                                timeout = min(5, CONST.MAX_DENDRITE_TIMEOUT),
                                deserialize=False,
                                run_async=True
                            )
                            any_success = any([r for r in responses if r.is_success])
                            if not any_success: #don't penalize the entire battery, just exit
                                bt.logging.error("\033[1;31mRETRY FAILED - NO SUCCESSFUL RESPONSES\033[0m")
                                synapse_with_event.event.set()
                                continue

                        et = time.perf_counter()
                        bt.logging.trace(f"Miners responded with {len(responses)} responses in \033[1;32m{et-st:0.4f}\033[0m seconds")
                       
                        # Adjust the scores based on responses from miners.
                        rewards = get_rewards(num_recs=number_of_recs_desired,
                                              ground_truth=api_request,
                                              responses=responses, actions=self.user_actions)
                        
                        if not len(chosen_uids) == len(responses) == len(rewards):
                            bt.logging.error("MISMATCH in lengths of chosen_uids, responses and rewards")
                            synapse_with_event.event.set()
                            continue
                        
                        # Default - send top score to client
                        selected_rec = rewards.argmax()
                        good_indices = np.where(rewards > 0)[0]
                        if len(good_indices) > 0:
                            good_responses = [responses[i] for i in good_indices]
                            bt.logging.info(f"Filtered to {len(good_responses)} from {len(responses)} total responses")
                            top_k = await self.analyze_similar_requests(number_of_recs_desired, good_responses)
                            if top_k and 1==1: #Top score now pulled from top_k
                                winner = safe_random.sample(top_k, 1)[0]
                                selected_rec = responses.index(winner)
                                rewards[selected_rec] *= CONSENSUS_BONUS_MULTIPLIER
                                bt.logging.info(f"\033[1;32m Consensus miner: {winner.miner_uid} from {winner.models_used} awarded bonus - batch: {winner.site_key} \033[0m")
                        else:
                            bt.logging.error("\033[1;33mZERO rewards - no valid candidates in responses \033[0m")
                            synapse_with_event.event.set()
                            continue
                    
                        elected : BitrecsRequest = responses[selected_rec]
                        elected.context = ""
                        elected.user = ""
                        
                        bt.logging.info(f"\033[1;32mFINAL MINER: {elected.miner_uid} \033[0m")
                        bt.logging.info(f"\033[1;32mFINAL MODEL: {elected.models_used} \033[0m")
                        bt.logging.info(f"\033[1;32mFINAL RESULT: {elected} \033[0m")
                        bt.logging.info(f"\033[1;32mFINAL Batch Id: {elected.site_key} \033[0m")
                        bt.logging.info(f"\033[1;32mFINAL Final Score: {rewards[selected_rec]} \033[0m")
                        bt.logging.trace(f"\033[1;32mQueue Size: {API_QUEUE.qsize()} \033[0m")
                        
                        if len(elected.results) == 0:
                            bt.logging.error("FATAL - Elected response has no results")
                            synapse_with_event.event.set()
                            continue
                        
                        synapse_with_event.output_synapse = elected
                        # Mark the synapse as processed, API will then return to the client
                        synapse_with_event.event.set()
                        self.total_request_in_interval +=1
                    
                        bt.logging.info(f"Scored responses: {rewards}")
                        self.update_scores(rewards, chosen_uids)
                        loop = asyncio.get_event_loop()
                        loop.run_in_executor(None, log_miner_responses_to_sql, self.step, responses)
                        bt.logging.info(f"SQL logging submitted to thread pool - step {self.step}")
                        
                    else:
                        if not api_exclusive: #Regular validator loop  
                            bt.logging.info("Processing synthetic concurrent forward")
                            #self.loop.run_until_complete(self.concurrent_forward())
                            raise NotImplementedError("concurrent_forward not implemented")

                    if self.should_exit:
                        return
                  
                    if self.step >= 1:
                        self.sync()
                    
                    self.step += 1

                except Exception as e:
                    bt.logging.error(f"Main validator RUN loop exception: {e}")
                    if synapse_with_event and synapse_with_event.event:
                        bt.logging.error("API MISSED REQUEST - Marking synapse as processed due to exception")
                        synapse_with_event.event.set()
                    bt.logging.error(traceback.format_exc())
                    bt.logging.error("\033[31m Sleeping for 60 seconds ... \033[0m")
                    await asyncio.sleep(60)                   
                finally:
                    if api_enabled and api_exclusive:
                        bt.logging.info(f"API MODE - forward finished, ready for next request")                        
                    else:
                        bt.logging.info(f"LIMP MODE forward finished, sleep for {45} seconds")
                        await asyncio.sleep(45)

        except KeyboardInterrupt:
            bt.logging.info("Caught keyboard interrupt. Cleaning up...")
            if self.api_server:
                self.api_server.stop()
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.error(traceback.format_exc())
            if self.api_server:
                self.api_server.stop()

    async def run(self):
        """Initiates and manages the main loop for the validator on the Bitrecs subnet."""
        await self.main_loop()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=lambda: anyio.run(self.run), daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            
            if self.api_server:
                self.api_server.stop()
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        Also ensures API server is properly shutdown.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            if self.api_server:
                self.api_server.stop()
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        bt.logging.info(f"set_weights on chain start")       
        bt.logging.info(f"Scores: {self.scores}")       

        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
        
        if np.all(self.scores == 0):
            bt.logging.warning(
                f"Scores are all zero. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
            return

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN
        
        bt.logging.debug("norm", norm)
        
        # Compute raw_weights safely
        raw_weights = self.scores / norm         
        
        # Printing type of arr object
        bt.logging.debug("Array is of type: ", type(raw_weights))
        # Printing array dimensions (axes)
        bt.logging.debug("No. of dimensions: ", raw_weights.ndim)
        # Printing shape of array
        bt.logging.debug("Shape of array: ", raw_weights.shape)
        # Printing size (total number of elements) of array
        bt.logging.debug("Size of array: ", raw_weights.size)
        # Printing type of elements in array
        bt.logging.debug("Array stores elements of type: ", raw_weights.dtype)        
        bt.logging.debug("uids", str(self.metagraph.uids.tolist()))
        bt.logging.debug("raw_weights", str(raw_weights))
        
        # Process the raw weights to final_weights via subtensor limitations.
        try:
            (
                processed_weight_uids,
                processed_weights,
            ) = process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
        except Exception as e:
            bt.logging.error(f"process_weights_for_netuid function error: {e}")
            pass
            
        bt.logging.debug(f"processed_weight_uids {processed_weight_uids}")        
        bt.logging.debug(f"processed_weights {processed_weights}")

        # Convert to uint16 weights and uids.
        try:
            (
                uint_uids,
                uint_weights,
            ) = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
                        
            bt.logging.debug(f"uint_weights {uint_weights}")        
            bt.logging.debug(f"uint_uids {uint_uids}")

            # Log weights to wandb before chain update
            weights_dict = {str(uid): float(weight) for uid, weight in zip(uint_uids, uint_weights)}
            if self.config.wandb.enabled and self.wandb:
                self.wandb.log_weights(self.step, weights_dict)

        except Exception as e:
            bt.logging.error(f"convert_weights_and_uids_for_emit function error: {e}")
            pass

         # Set the weights on chain via our subtensor connection.
        try:
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )
            if result is True:
                bt.logging.info(f"set_weights on chain successfully! msg: {msg}")
                if self.config.wandb.enabled and self.wandb:
                    self.wandb.log_metrics({"weight_update_success": 1})
            else:
                bt.logging.error(f"set_weights on chain failed {msg}")
                if self.config.wandb.enabled and self.wandb:
                    self.wandb.log_metrics({"weight_update_success": 0})
        except Exception as e:
            bt.logging.error(f"set_weights failed with exception: {e}")


    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        #bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")
        

    def save_state(self):
        if self.first_sync:
            bt.logging.info(f"Save State - first_sync = True, skipping save state.")
            self.first_sync = False
            return
        
        if (np.all(self.scores == 0) or self.scores.size == 0):
            bt.logging.warning(f"Score array is empty or all zeros. Skipping save state.")
            return

        state_path = self.config.neuron.full_path + "/state.npz"        
        np.savez(
            state_path,
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys
        )
        if os.path.isfile(state_path):
            bt.logging.info(f"\033[32m Save state confirmed \033[0m")
        else:
            bt.logging.error(f"Save state failed.")
            raise Exception("Save state failed, file not found after save attempt.")


    def load_state(self):
        state_path = self.config.neuron.full_path + "/state.npz"
        try:
            if not os.path.exists(state_path):
                bt.logging.info("No state found - initializing first step")
                self.step = 0
                return

            state = np.load(state_path, allow_pickle=True)
            self.step = int(state["step"])
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
          
            bt.logging.trace(f"State loaded at step {self.step}")
                
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")
            self.step = 0


