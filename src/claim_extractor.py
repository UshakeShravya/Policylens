import re
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

    results = extract_claims(sample_pages)

    print(f"Extracted {len(results)} claims\n")
    for c in results:
        print(f"[{c['claim_id']}] (p.{c['page_number']}) flags={c['flags']}")
        print(f"     {c['text']}")
        if c["raw_entities"]:
            entities = ", ".join(f"{e['text']} ({e['label']})" for e in c["raw_entities"])
            print(f"     entities: {entities}")
        print()
