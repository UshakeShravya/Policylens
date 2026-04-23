"""
Agent layer: wraps the deterministic verifier with a Claude tool-calling loop.

The agent has access to three tools per claim investigation:
  retrieve_evidence  — semantic search over the FAISS index
  run_rule_checks    — deterministic 4-rule engine (threshold, numeric, causal, entity)
  get_page_context   — all chunks from a given page

Claude orchestrates those tools, then returns a structured JSON verdict that may
complement or override the rule-based result.  Falls back to the deterministic
engine transparently when ANTHROPIC_API_KEY is not set or the API is unavailable.
"""

import concurrent.futures
import json
import os
import re
import sys
import warnings

_MODEL = "claude-sonnet-4-6"
_MAX_TURNS = 6


def _log(msg: str) -> None:
    """Write a timestamped agent debug line to stderr (always visible in the terminal)."""
    print(f"[AGENT] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Prompts & tool definitions
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are a policy document verification expert auditing claims for hallucinations.

Your task: decide whether a specific claim is supported by the source policy document.

You have three tools:
  retrieve_evidence  — semantic search; returns ranked passages with similarity scores
  run_rule_checks    — deterministic engine: similarity threshold (0.35), numeric match
                       (±10%), causal scrutiny, entity consistency
  get_page_context   — all document chunks from a given page number

Recommended workflow:
1. Call retrieve_evidence with the claim text (top_k=3 to start).
2. Call run_rule_checks with those chunk_ids to get the baseline verdict.
3. If the verdict is borderline, or the claim has numeric/causal flags, retrieve
   more context: broader top_k, different query, or the source page directly.
4. Reason carefully: does the evidence actually assert what the claim states?
   Pay attention to hedging language ("may have", "associated with") vs. direct
   causation, and to whether specific numbers appear verbatim in the source.

When you are done investigating, respond with ONLY a JSON object:
{
  "verdict":          "Supported" | "Partially Supported" | "Unsupported" | "High-Risk Silent Failure",
  "confidence_score": float between 0.0 and 1.0,
  "reasoning":        "Concise explanation referencing specific evidence",
  "agent_notes":      "Any observation beyond rule-based logic (may be empty string)"
}
Do not include any text outside the JSON object in your final response.\
"""

_TOOLS = [
    {
        "name": "retrieve_evidence",
        "description": (
            "Search the policy document for passages semantically similar to a query. "
            "Returns the top-k most relevant chunks with similarity scores and page numbers. "
            "Use this to discover what the source document actually says about a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or claim to search for in the document",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (1–10); default 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_rule_checks",
        "description": (
            "Apply the 4-rule deterministic verification engine to the claim using "
            "previously retrieved evidence chunks.  Rules: (1) similarity threshold ≥ 0.35, "
            "(2) numeric match ±10%, (3) causal scrutiny, (4) entity consistency. "
            "chunk_ids must come from a prior retrieve_evidence call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "chunk_id values from retrieve_evidence results",
                },
            },
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "get_page_context",
        "description": (
            "Retrieve all document chunks from a specific page number. "
            "Useful for broader context when a claim references a particular page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page_number": {
                    "type": "integer",
                    "description": "The page number whose chunks to return",
                },
            },
            "required": ["page_number"],
        },
        # FIX: cache_control belongs at the TOOL level, not inside input_schema.
        # Placing it inside input_schema was silently ignored at best, and caused
        # API validation errors at worst — in either case no caching occurred.
        "cache_control": {"type": "ephemeral"},
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_agent_json(text: str) -> dict | None:
    """Extract the JSON verdict object from Claude's final text response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _serialize_content(content_blocks) -> list[dict]:
    """
    Convert SDK ContentBlock objects to plain dicts safe for re-submission.

    The Anthropic SDK returns Pydantic model instances (TextBlock, ToolUseBlock).
    Passing them directly back into messages works in some SDK versions but not
    all — explicit serialization avoids subtle cross-thread / version issues.
    """
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        # other block types (e.g. thinking) are dropped intentionally
    return result


def _run_agent_loop(
    client,
    claim: dict,
    index,
    chunks: list[dict],
) -> dict | None:
    """
    Execute the tool-calling loop for a single claim.
    Returns the parsed verdict dict, or None if the loop exhausts max turns
    or Claude never produces a valid JSON verdict.
    """
    from src.retriever import retrieve_evidence as _retrieve
    from src.verifier import verify_claim as _verify

    chunk_by_id: dict[int, dict] = {c["chunk_id"]: c for c in chunks}
    evidence_cache: dict[int, dict] = {}

    def _dispatch(name: str, inputs: dict):
        if name == "retrieve_evidence":
            query = inputs.get("query", claim["text"])
            top_k = min(int(inputs.get("top_k", 3)), 10)
            results = _retrieve(query, index, chunks, top_k=top_k)
            for r in results:
                evidence_cache[r["chunk_id"]] = r
            return results

        if name == "run_rule_checks":
            chunk_ids = inputs.get("chunk_ids", [])
            evidence = []
            missing = []
            for cid in chunk_ids:
                if cid in evidence_cache:
                    evidence.append(evidence_cache[cid])
                elif cid in chunk_by_id:
                    evidence.append({**chunk_by_id[cid], "similarity_score": 0.0})
                else:
                    missing.append(cid)
            if not evidence:
                return {"error": "No valid chunk_ids — call retrieve_evidence first"}
            r = _verify(claim, evidence)
            result = {
                "verdict": r["verdict"],
                "rules_triggered": r["rules_triggered"],
                "risk_explanation": r["risk_explanation"],
                "confidence_score": r["confidence_score"],
            }
            if missing:
                result["warning"] = f"chunk_ids not found: {missing}"
            return result

        if name == "get_page_context":
            page_number = int(inputs.get("page_number", claim["page_number"]))
            return [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "page_number": c["page_number"],
                }
                for c in chunks
                if c["page_number"] == page_number
            ]

        return {"error": f"Unknown tool: {name}"}

    user_message = (
        f"Claim to verify (ID {claim['claim_id']}):\n"
        f'"{claim["text"]}"\n\n'
        f"Source page: {claim['page_number']}\n"
        f"Flags: {claim['flags']}\n"
        f"Named entities: {[e['text'] for e in claim.get('raw_entities', [])]}\n\n"
        "Investigate this claim using the available tools, then return your verdict "
        "as a JSON object."
    )

    messages: list[dict] = [{"role": "user", "content": user_message}]

    _log(f"claim={claim['claim_id']} starting loop | {claim['text'][:70]}")

    for turn in range(_MAX_TURNS):
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _AGENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=_TOOLS,
            messages=messages,
        )

        _log(f"claim={claim['claim_id']} turn={turn + 1} stop_reason={response.stop_reason}")

        # Serialize to plain dicts before re-submitting (avoids SDK version quirks)
        asst_content = _serialize_content(response.content)
        messages.append({"role": "assistant", "content": asst_content})

        if response.stop_reason == "end_turn":
            text = next((b["text"] for b in asst_content if b["type"] == "text"), "")
            _log(f"claim={claim['claim_id']} final text preview: {text[:120]}")
            verdict = _parse_agent_json(text)
            if verdict is None:
                _log(f"claim={claim['claim_id']} WARNING: could not parse JSON from final response")
            return verdict

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in asst_content:
                if block["type"] == "tool_use":
                    name = block["name"]
                    inputs = block["input"]
                    _log(f"claim={claim['claim_id']} → tool={name} input={str(inputs)[:100]}")
                    result = _dispatch(name, inputs)
                    result_preview = str(result)[:100]
                    _log(f"claim={claim['claim_id']} ← result={result_preview}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(result, default=str),
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            continue

        _log(f"claim={claim['claim_id']} WARNING: unexpected stop_reason={response.stop_reason} — ending loop")
        break

    _log(f"claim={claim['claim_id']} loop ended without final verdict (turns exhausted or unexpected stop)")
    return None


def _deterministic_fallback(claim: dict, index, chunks: list[dict]) -> dict:
    from src.retriever import retrieve_evidence
    from src.verifier import verify_claim

    evidence = retrieve_evidence(claim["text"], index, chunks, top_k=3)
    result = verify_claim(claim, evidence)
    result["agent_notes"] = ""
    return result


def _build_verdict(
    agent_result: dict,
    claim: dict,
    index,
    chunks: list[dict],
) -> dict:
    """Merge an agent JSON result with the deterministic base verdict."""
    from src.retriever import retrieve_evidence
    from src.verifier import verify_claim

    evidence = retrieve_evidence(claim["text"], index, chunks, top_k=3)
    base = verify_claim(claim, evidence)

    valid_verdicts = {
        "Supported",
        "Partially Supported",
        "Unsupported",
        "High-Risk Silent Failure",
    }
    verdict = agent_result.get("verdict", base["verdict"])
    if verdict not in valid_verdicts:
        verdict = base["verdict"]

    confidence = agent_result.get("confidence_score", base["confidence_score"])
    try:
        confidence = float(confidence)
        confidence = max(0.01, min(0.99, confidence))
    except (TypeError, ValueError):
        confidence = base["confidence_score"]

    return {
        **base,
        "verdict": verdict,
        "confidence_score": round(confidence, 4),
        "risk_explanation": agent_result.get("reasoning", base["risk_explanation"]),
        "agent_notes": agent_result.get("agent_notes", ""),
        "detection_method": claim.get("detection_method", "regex") + "+agent",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claim_with_agent(claim: dict, index, chunks: list[dict]) -> dict:
    """
    Verify a single claim using the Claude agent loop.

    Returns a verdict dict identical to verifier.verify_claim but with two
    extra fields:
      agent_notes      : str — Claude's nuanced observations
      detection_method : original value suffixed with "+agent"

    Falls back to the deterministic engine (with a visible error line on stderr)
    if ANTHROPIC_API_KEY is not set, the package is missing, or the API fails.
    """
    try:
        import anthropic
    except ImportError:
        _log("anthropic package not installed — deterministic fallback")
        return _deterministic_fallback(claim, index, chunks)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        _log("ANTHROPIC_API_KEY not set — deterministic fallback")
        return _deterministic_fallback(claim, index, chunks)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        agent_result = _run_agent_loop(client, claim, index, chunks)
    except Exception as exc:
        # Surface the actual error — credit exhaustion, rate limits, network issues
        # all appear here and were previously invisible.
        _log(f"claim={claim['claim_id']} API error ({type(exc).__name__}): {exc}")
        return _deterministic_fallback(claim, index, chunks)

    if agent_result is None:
        _log(f"claim={claim['claim_id']} loop returned no verdict — deterministic fallback")
        return _deterministic_fallback(claim, index, chunks)

    _log(f"claim={claim['claim_id']} agent verdict={agent_result.get('verdict')} confidence={agent_result.get('confidence_score')}")
    return _build_verdict(agent_result, claim, index, chunks)


def verify_claims_with_agent(
    claims: list[dict],
    index,
    chunks: list[dict],
) -> list[dict]:
    """
    Run agent-based verification for a list of claims in parallel (max 3 workers).
    Drop-in replacement for verifier.verify_claims; preserves claim order.
    Falls back to the deterministic engine per-claim on any failure.
    """
    try:
        import anthropic
    except ImportError:
        _log("anthropic package not installed — deterministic fallback for all claims")
        return [_deterministic_fallback(c, index, chunks) for c in claims]

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        _log("ANTHROPIC_API_KEY not set — deterministic fallback for all claims")
        return [_deterministic_fallback(c, index, chunks) for c in claims]

    client = anthropic.Anthropic(api_key=api_key)

    def _run_one(claim: dict) -> dict:
        try:
            agent_result = _run_agent_loop(client, claim, index, chunks)
        except Exception as exc:
            _log(f"claim={claim['claim_id']} API error ({type(exc).__name__}): {exc}")
            return _deterministic_fallback(claim, index, chunks)

        if agent_result is None:
            _log(f"claim={claim['claim_id']} loop returned no verdict — deterministic fallback")
            return _deterministic_fallback(claim, index, chunks)

        _log(f"claim={claim['claim_id']} agent verdict={agent_result.get('verdict')}")
        return _build_verdict(agent_result, claim, index, chunks)

    results: list[dict | None] = [None] * len(claims)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_idx = {
            pool.submit(_run_one, claim): i for i, claim in enumerate(claims)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                _log(f"claim={claims[idx]['claim_id']} worker failed ({type(exc).__name__}): {exc}")
                results[idx] = _deterministic_fallback(claims[idx], index, chunks)

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Standalone diagnostic test  — all output to stdout, no stderr
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    def p(msg=""):
        print(msg, flush=True)

    p("=" * 64)
    p("  PolicyLens agent diagnostic")
    p("=" * 64)

    # ── Step 0: load env & check key ────────────────────────────────────
    p("\n[0] Environment")
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(override=True)
        p(f"  load_dotenv: {'found .env' if loaded else 'no .env file found'}")
    except ImportError:
        p("  python-dotenv not installed")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        p(f"  ANTHROPIC_API_KEY: {api_key[:18]}...  (len={len(api_key)})")
    else:
        p("  ANTHROPIC_API_KEY: NOT SET — cannot proceed")
        raise SystemExit(1)

    # ── Step 1: raw API call (no tools) ─────────────────────────────────
    p("\n[1] Raw API connectivity check (no tools)")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        ping = client.messages.create(
            model=_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
        )
        p(f"  status: OK")
        p(f"  response: {ping.content[0].text!r}")
        p(f"  stop_reason: {ping.stop_reason}")
    except Exception as exc:
        p(f"  FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    # ── Step 2: build minimal test fixture ──────────────────────────────
    p("\n[2] Building test index")
    from src.retriever import chunk_pages, build_index

    SOURCE_PAGES = [
        {
            "page_number": 1,
            "text": (
                "Emissions declined during the review period covering 2018 to 2022, "
                "though macroeconomic factors may have contributed to the trend. "
                "The Environmental Protection Agency monitored air quality across 12 states. "
                "No single cause was identified for the observed reduction."
            ),
        },
    ]
    chunks = chunk_pages(SOURCE_PAGES, chunk_size=3)
    index, chunks = build_index(chunks)
    p(f"  chunks: {len(chunks)}")

    TEST_CLAIM = {
        "claim_id": 99,
        "text": "The policy reduced emissions by 22% over five years.",
        "page_number": 1,
        "flags": ["numeric", "causal_verb"],
        "raw_entities": [],
        "detection_method": "regex",
    }

    # ── Step 3: run the agent loop with live turn-by-turn output ────────
    p("\n[3] Running agent loop (tool calls will print below)")
    p("-" * 64)

    # Temporarily redirect _log to stdout for this test
    import sys as _sys
    original_log_target = _sys.stderr

    def _stdout_log(msg: str):
        print(f"[AGENT] {msg}", flush=True)

    # Monkey-patch _log to stdout just for this test
    import src.agent as _self
    _original_log = _self._log
    _self._log = _stdout_log

    try:
        result = verify_claim_with_agent(TEST_CLAIM, index, chunks)
    except Exception as exc:
        p(f"\nEXCEPTION in verify_claim_with_agent:")
        traceback.print_exc()
        result = None
    finally:
        _self._log = _original_log  # restore

    p("-" * 64)

    # ── Step 4: print full result ────────────────────────────────────────
    p("\n[4] Full result")
    if result is None:
        p("  result: None (agent returned nothing)")
    else:
        p(f"  verdict:          {result['verdict']}")
        p(f"  confidence:       {result['confidence_score']}")
        p(f"  detection_method: {result['detection_method']}")
        p(f"  agent_notes:      {result.get('agent_notes') or '(empty)'}")
        p(f"  risk_explanation: {result['risk_explanation'][:120]}")
        p()
        if "+agent" in result.get("detection_method", ""):
            p("  PASS — agent loop ran successfully")
        else:
            p("  FAIL — fell back to deterministic (detection_method lacks '+agent')")
