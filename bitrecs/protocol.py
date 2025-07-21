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


import pydantic
import bittensor as bt

class BitrecsRequest(bt.Synapse):
    created_at: str | None = pydantic.Field(
        None,
        description="Timestamp of the request creation"
    )
    user: str | None = pydantic.Field(
        None,
        description="Ecommerce profile for the request"
    )
    num_results: int = pydantic.Field(
        0,
        description="Expected number of recs",
    )
    query: str | None = pydantic.Field(
        None,
        description="The query sku",
    )
    context: str | None = pydantic.Field(
        None,
        description="The store catalog as a json array",
    )    
    site_key: str | None = pydantic.Field(
        None,
        description="Batch ID for grouping requests (this is not a sensitive value)"
    )
    results: list | None = pydantic.Field(
        None,
        description="List of miner recommendations"
    )    
    models_used: list | None = pydantic.Field(
        None,
        description="List of miner models used to generate the results (exepect 1 value only)"
    )
    miner_uid: str | None = pydantic.Field(
        None,
        description="Miner UID"
    )
    miner_hotkey: str | None = pydantic.Field(
        None,
        description="Miner Hotkey"
    )
    miner_signature: str | None = pydantic.Field(
        None,
        description="Signature of the miner's hotkey over the payload",
    )
    

    def to_dict(self) -> dict:
        return {
            'created_at': self.created_at,
            'user': self.user,
            'num_results': self.num_results,
            'query': self.query,
            'context': self.context,
            'site_key': self.site_key,
            'results': str(self.results) if self.results else None,
            'models_used': str(self.models_used) if self.models_used else None,
            'miner_uid': self.miner_uid,
            'miner_hotkey': self.miner_hotkey,
            'miner_signature': self.miner_signature,
        }