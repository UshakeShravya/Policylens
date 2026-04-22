import json
import os
import re
import warnings
from difflib import SequenceMatcher

import spacy

_NUMERIC_RE = re.compile(
    r"""
    \b\d[\d,]*\.?\d*\s*%          # percentages: 22%, 3.5%
    | \b\d[\d,]*\.?\d*\s*         # plain numbers: 3.5, 1,000
      (?:million|billion|trillion|thousand|hundred)?\b
    | \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten
          |eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen
          |eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy
          |eighty|ninety|hundred|thousand|million|billion)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CAUSAL_RE = re.compile(
    r"\b(?:caused?|leads?|led\s+to|results?(?:ed)?\s+in|reduced?|increases?|"
    r"increased?|decreased?|improved?|generated?|produced?|triggered?|"
    r"attributed?\s+to|contributed?\s+to|drove|driven\s+by|due\s+to|"
    r"owing\s+to|because(?:\s+of)?|consequently|therefore|thus|hence)\b",
    re.IGNORECASE,
)

_SUPERLATIVE_RE = re.compile(
    r"\b(?:highest?|lowest?|greatest?|least|most|fewest?|best|worst|"
    r"largest?|smallest?|all|every|never|always|none|entirely|completely|"
    r"absolutely|consistently|unprecedented|record-(?:high|low))\b",
    re.IGNORECASE,
)

# spaCy entity types worth tracking for policy documents
_ENTITY_TYPES = {"ORG", "GPE", "LOC", "PERSON", "DATE", "PERCENT", "MONEY", "QUANTITY", "LAW"}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _flag_sentence(sent_text: str, doc_sent) -> tuple[list[str], list[dict]]:
    """
    Return (flags, raw_entities) for a single spaCy sentence.

    flags is a list of zero or more strings from:
        "numeric", "causal_verb", "named_entity", "superlative"
    raw_entities is a list of {"text": str, "label": str} dicts.
    """
    flags = []

    if _NUMERIC_RE.search(sent_text):
        flags.append("numeric")

    if _CAUSAL_RE.search(sent_text):
        flags.append("causal_verb")

    if _SUPERLATIVE_RE.search(sent_text):
        flags.append("superlative")

    raw_entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc_sent.ents
        if ent.label_ in _ENTITY_TYPES
    ]
    if raw_entities:
        flags.append("named_entity")

    return flags, raw_entities


def extract_claims(pages: list[dict]) -> list[dict]:
    """
    Extract verifiable factual claims from parsed PDF pages.

    Parameters
    ----------
    pages : list[dict]
        Output of parser.extract_text_from_pdf — each dict has
        "page_number" (int) and "text" (str).

    Returns
    -------
    list[dict]
        One dict per flagged sentence:
        {
            "claim_id":     int,          # sequential, 1-indexed
            "text":         str,          # raw sentence text
            "page_number":  int,          # source page
            "flags":        list[str],    # why it was flagged
            "raw_entities": list[dict],   # spaCy named entities
        }
    """
    nlp = _get_nlp()
    claims = []
    claim_id = 1

    for page in pages:
        page_num = page["page_number"]
        text = page["text"]

        # spaCy handles both sentence segmentation and NER in one pass
        doc = nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Skip very short fragments — not real claims
            if len(sent_text.split()) < 5:
                continue

            flags, raw_entities = _flag_sentence(sent_text, sent.as_doc())

            if not flags:
                continue

            claims.append({
                "claim_id": claim_id,
                "text": sent_text,
                "page_number": page_num,
                "flags": flags,
                "raw_entities": raw_entities,
            })
            claim_id += 1

    return claims


_LLM_SYSTEM_PROMPT = """\
You are a policy document auditor specialising in hallucination detection.

Your task: read the policy document page text provided and return EVERY sentence
that contains a verifiable factual claim. A verifiable factual claim is any
statement that:
  - includes a specific number, percentage, dollar amount, or quantity
  - names an organisation, law, person, or geographic entity in a factual context
  - asserts causation (e.g. "led to", "reduced", "generated", "resulted in")
  - uses superlative or absolute language (e.g. "highest", "all", "never", "unprecedented")

Return ONLY a JSON array of objects. Each object must have exactly these fields:
  "text"  : the full verbatim sentence text
  "flags" : array of applicable flag strings from
            ["numeric","causal_verb","named_entity","superlative"]

If there are no verifiable claims on the page, return an empty array: []

Do NOT include explanations, markdown prose, or any text outside the JSON array.\
"""

_SIMILARITY_THRESHOLD = 0.85  # SequenceMatcher ratio above which two claims are duplicates


def _parse_llm_json(raw: str) -> list[dict]:
    """Strip optional markdown fences and parse JSON from Claude's response."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return json.loads(text)


def _are_duplicate(text_a: str, text_b: str) -> bool:
    ratio = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
    return ratio >= _SIMILARITY_THRESHOLD


def extract_claims_with_llm(pages: list[dict]) -> list[dict]:
    """
    Extract verifiable factual claims using regex/spaCy followed by a Claude
    API second-pass for each page.

    Parameters
    ----------
    pages : list[dict]
        Output of parser.extract_text_from_pdf.

    Returns
    -------
    list[dict]
        Same schema as extract_claims() plus a "detection_method" field:
        "regex", "llm", or "both".

    Falls back to extract_claims() (without LLM) if ANTHROPIC_API_KEY is not set.
    """
    # Lazy import so the rest of the module works without anthropic installed
    try:
        import anthropic
    except ImportError:
        warnings.warn(
            "anthropic package not installed — falling back to regex/spaCy extraction.",
            stacklevel=2,
        )
        claims = extract_claims(pages)
        for c in claims:
            c["detection_method"] = "regex"
        return claims

    # Load API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv optional; key may already be in environment

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        warnings.warn(
            "ANTHROPIC_API_KEY not set — falling back to regex/spaCy extraction.",
            stacklevel=2,
        )
        claims = extract_claims(pages)
        for c in claims:
            c["detection_method"] = "regex"
        return claims

    # --- Regex/spaCy first pass ---
    regex_claims = extract_claims(pages)

    # Index regex claims by page for fast lookup during merge
    regex_by_page: dict[int, list[dict]] = {}
    for c in regex_claims:
        regex_by_page.setdefault(c["page_number"], []).append(c)

    client = anthropic.Anthropic(api_key=api_key)

    merged: list[dict] = []
    # Start claim_id after all regex claims so IDs remain unique
    next_id = (max((c["claim_id"] for c in regex_claims), default=0) + 1)

    # Track which regex claims were matched by the LLM (to mark them "both")
    matched_regex_ids: set[int] = set()

    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"].strip()
        if not page_text:
            continue

        # --- LLM second pass with prompt caching on the system block ---
        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": _LLM_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"Page {page_num}:\n\n{page_text}",
                    }
                ],
            )
            raw = response.content[0].text
            llm_items = _parse_llm_json(raw)
        except Exception as exc:
            warnings.warn(
                f"LLM extraction failed for page {page_num}: {exc}. "
                "Using regex results for this page.",
                stacklevel=2,
            )
            llm_items = []

        page_regex = regex_by_page.get(page_num, [])

        for item in llm_items:
            llm_text = item.get("text", "").strip()
            if not llm_text or len(llm_text.split()) < 5:
                continue
            llm_flags = item.get("flags", [])

            # Check if this LLM claim matches an existing regex claim
            match = None
            for rc in page_regex:
                if _are_duplicate(llm_text, rc["text"]):
                    match = rc
                    break

            if match is not None:
                matched_regex_ids.add(match["claim_id"])
                # Merge: use regex text/entities; union the flags
                merged_flags = list(dict.fromkeys(match["flags"] + llm_flags))
                merged.append({
                    "claim_id": match["claim_id"],
                    "text": match["text"],
                    "page_number": page_num,
                    "flags": merged_flags,
                    "raw_entities": match["raw_entities"],
                    "detection_method": "both",
                })
            else:
                # LLM-only discovery
                merged.append({
                    "claim_id": next_id,
                    "text": llm_text,
                    "page_number": page_num,
                    "flags": llm_flags,
                    "raw_entities": [],
                    "detection_method": "llm",
                })
                next_id += 1

    # Add regex-only claims (not matched by LLM) at their original IDs
    regex_only = [
        {**c, "detection_method": "regex"}
        for c in regex_claims
        if c["claim_id"] not in matched_regex_ids
    ]
    merged.extend(regex_only)

    # Sort by (page_number, claim_id) for a stable, readable order
    merged.sort(key=lambda c: (c["page_number"], c["claim_id"]))

    return merged


if __name__ == "__main__":
    sample_pages = [
        {
            "page_number": 1,
            "text": (
                "The review period covered fiscal years 2018 through 2022. "
                "Emissions declined during the review period, though macroeconomic "
                "factors may have contributed. "
                "The policy reduced emissions by 22% over five years. "
                "The Environmental Protection Agency confirmed the findings. "
                "This represents the highest recorded decline in the sector. "
                "Weather was pleasant."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "The program generated 3.5 million jobs across the Midwest. "
                "Unemployment never exceeded 4% during the implementation phase. "
                "Congress passed the Inflation Reduction Act in August 2022. "
                "All participating states reported improved outcomes. "
                "Results were positive."
            ),
        },
    ]

    def _print_claims(claims: list[dict], label: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {label}  —  {len(claims)} claim(s) extracted")
        print("=" * 60)
        for c in claims:
            method = c.get("detection_method", "—")
            print(
                f"[{c['claim_id']}] (p.{c['page_number']}) "
                f"flags={c['flags']}  method={method}"
            )
            print(f"     {c['text']}")
            if c.get("raw_entities"):
                entities = ", ".join(
                    f"{e['text']} ({e['label']})" for e in c["raw_entities"]
                )
                print(f"     entities: {entities}")
            print()

    # --- Pass 1: regex/spaCy only ---
    regex_results = extract_claims(sample_pages)
    _print_claims(regex_results, "Regex / spaCy pass")

    # --- Pass 2: LLM-assisted (falls back gracefully if key missing) ---
    llm_results = extract_claims_with_llm(sample_pages)
    _print_claims(llm_results, "LLM-assisted pass (regex + Claude)")
