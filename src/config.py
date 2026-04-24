"""
Central configuration for PolicyLens.

All model IDs, retry parameters, and global limits live here.
Update this file when Anthropic releases new model versions.
"""

# ---------------------------------------------------------------------------
# Anthropic model IDs
# ---------------------------------------------------------------------------

# Used by claim_extractor for LLM-assisted text extraction
EXTRACTION_MODEL: str = "claude-opus-4-7"

# Used by claim_extractor for multimodal (Vision) extraction
VISION_MODEL: str = "claude-sonnet-4-6"

# Used by the agent orchestration loop
AGENT_MODEL: str = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# API retry policy (applied to every Anthropic API call)
# ---------------------------------------------------------------------------

# Maximum number of retry attempts on transient errors (429, 529, network)
API_MAX_RETRIES: int = 3

# Base delay in seconds; doubles on each attempt (exponential backoff)
API_RETRY_BASE_S: float = 2.0

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

# Maximum tool-calling turns per claim before giving up
AGENT_MAX_TURNS: int = 6

# Maximum concurrent agent workers across claims
AGENT_MAX_WORKERS: int = 3

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

# sentence-transformers model for chunk embeddings
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# App limits
# ---------------------------------------------------------------------------

# Maximum PDF upload size in megabytes
MAX_UPLOAD_MB: int = 50

# Maximum pages passed to the Vision API for multimodal claim extraction.
# Charts and figures are densest in the first ~20 pages of policy documents;
# scanning the full document would cost hundreds of API calls on large PDFs.
MULTIMODAL_MAX_PAGES: int = 20
