from bitrecs.protocol import BitrecsRequest
from bitrecs.utils import constants as CONST


def validate_br_request(synapse: BitrecsRequest) -> bool:    
    return (
        isinstance(synapse, BitrecsRequest) and
        synapse.query and synapse.context and synapse.site_key and
        CONST.MIN_QUERY_LENGTH <= len(synapse.query) <= CONST.MAX_QUERY_LENGTH and
        len(synapse.context) <= CONST.MAX_CONTEXT_TEXT_LENGTH and
        1 <= synapse.num_results <= CONST.MAX_RECS_PER_REQUEST and
        len(synapse.results) == 0 and
        len(synapse.models_used) == 0
    )