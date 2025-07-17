import re
import bitrecs
import datetime
from pathlib import Path
from datetime import datetime, timezone

"""
Global constants

Constants:
    ROOT_DIR (Path): Root directory of the project.
    SCHEMA_UPDATE_CUTOFF (datetime): Cutoff date for schema updates.
    MAX_DENDRITE_TIMEOUT (int): Length of seconds given to miners to respond to a dendrite request.
    MIN_QUERY_LENGTH (int): Minimum length of a query.
    MAX_QUERY_LENGTH (int): Maximum length of a query.
    MAX_RECS_PER_REQUEST (int): Maximum number of recommendations per request.
    MAX_CONTEXT_LENGTH (int): Maximum length of a context.
    MIN_CATALOG_SIZE (int): Minimum size of a request catalog.
    MAX_CATALOG_SIZE (int): Maximum size of a request catalog.
    MINER_BATTERY_INTERVAL (int): Length of seconds between miner checks.
    MINER_BATCH_SIZE (int): Number of miners to check in a single ping batch.
    ACTION_SYNC_INTERVAL (int): Length of seconds between action syncs.
    VERSION_CHECK_INTERVAL (int): Length of seconds between version checks.
    COOLDOWN_SYNC_INTERVAL (int): Length of seconds between cooldown syncs.
    CATALOG_DUPE_THRESHOLD (float): Threshold for duplicate products in a catalog.
    R2_SYNC_INTERVAL (int): Length of seconds between R2 syncs.
    RE_PRODUCT_NAME (Pattern): Regular expression to clean product names.
    RE_REASON (Pattern): Regular expression to clean reasons.
    RE_MODEL_NAME (Pattern): Regular expression to clean model names.
    CONVERSION_SCORING_ENABLED (bool): Flag to enable conversion scoring.
    MIN_ACTIVE_MINERS (int): Minimum number of active miners required.
    MAX_MINER_FILL_ATTEMPTS (int): Maximum number of attempts for miner battery loading.
    SCORE_DISPLAY_ENABLED (bool): Flag to enable scoring metrics display.
    SCORE_DISPLAY_INTERVAL (int): Length of seconds between metrics displays.
    EPOCH_TEMPO (int): Number of blocks in an epoch.

"""

ROOT_DIR = Path(bitrecs.__file__).parent.parent
SCHEMA_UPDATE_CUTOFF = datetime(2025, 7, 21, tzinfo=timezone.utc)
MAX_DENDRITE_TIMEOUT = 5
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 40
MIN_RECS_PER_REQUEST = 1
MAX_RECS_PER_REQUEST = 20
MAX_CONTEXT_TEXT_LENGTH = 1_000_000
MAX_CONTEXT_TOKEN_COUNT = 600_000
MIN_CATALOG_SIZE = 6
MAX_CATALOG_SIZE = 100_000
MINER_BATTERY_INTERVAL = 300
MINER_BATCH_SIZE = 4
ACTION_SYNC_INTERVAL = 14400
VERSION_CHECK_INTERVAL = 1200
COOLDOWN_SYNC_INTERVAL = 360
CATALOG_DUPE_THRESHOLD = 0.05
R2_SYNC_INTERVAL = 3600
RE_PRODUCT_NAME = re.compile(r"[^A-Za-z0-9 |-]")
RE_REASON = re.compile(r"[^A-Za-z0-9 ]")
RE_MODEL_NAME = re.compile(r"[^A-Za-z0-9-._/-:]")
CONVERSION_SCORING_ENABLED = False
MIN_ACTIVE_MINERS = 7
MAX_MINER_FILL_ATTEMPTS = 5
SCORE_DISPLAY_ENABLED = True
SCORE_DISPLAY_INTERVAL = 180
EPOCH_TEMPO = 360