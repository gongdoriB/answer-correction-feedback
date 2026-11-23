"""Extract whole source paragraphs, never generate or certify an explanation.

Selection uses source text and gold agreement only, never Qwen follow-up outcomes.
One contiguous paragraph avoids stitching unrelated premises/conclusions together.
"""
import re

POLICY = {
    "version": "contiguous-paragraphs-1",
    "source": "author-released GPT-4o (2024-05-13), five-shot predictions",
    "caps_in_order": [2, 3, 4],
    "threshold": 100,
    "threshold_denominator": "all 300 original questions, not only initial errors",
    "selection": "latest eligible ending paragraph, then longest contiguous block within cap; cap selected by availability before new inference",
    "limits": "Mechanical completeness/support-cue filters are not semantic or logical validation. Gold agreement validates final label only.",
    "word_range": [15, 180],
    "equations": "Preserve equation text; an equation-only paragraph counts as one extra unit and needs surrounding prose.",
    "sentences": "Terminal .?! outside TeX spans; decimal/abbreviation/list periods protected; unpunctuated nonempty tail counts as a sentence.",
    "no_rewriting": "Exact contiguous source paragraphs, whitespace only trimmed. No summaries, label deletion, or new reasons.",
}
MATH = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$", re.S)
LABEL = re.compile(r"\b(?:option|choice|answer)\s*(?:is\s*|[:=]\s*)?[(*]*[A-J]\b|(?m:^[ \t]*(?:[-*]\s*)?\**[A-J][.)]\s)", re.I)
SUPPORT = re.compile(r"\b(?:because|since|means|defined|definition|refers|indicates|implies|formula|law|principle|proportional|corresponds|results? in|leads? to|equal|equation|calculat\w*|differen\w*|probability|ratio|requires?|involves?|represents?)\b|=|\\frac", re.I)
META = re.compile(r"^(?:to (?:determine|find|solve|answer|understand)|let(?:'s| us)|we need to|given the (?:options|analysis)|based on (?:the|this) (?:analysis|evaluation))\b", re.I)


def sentence_count(text):
    masked = MATH.sub(" MATH ", text)
    masked = re.sub(r"(?<=\d)\.(?=\d)", "DOT", masked)
    masked = re.sub(r"\b(?:e\.g|i\.e|Dr|Mr|Mrs|vs|approx)\.", "ABBR", masked, flags=re.I)
    masked = re.sub(r"(?m)^\s*\d+\.\s", "", masked)
    parts = [p for p in re.split(r"[.!?]+(?:[\"'*\])]*)(?:\s+|$)", masked) if re.search(r"[A-Za-z0-9]", p)]
    return len(parts)


def paragraph_candidates(response, cap):
    found = []
    paragraphs = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", response, re.S))
    for end in range(len(paragraphs)):
      for start in range(end, -1, -1):
        text = response[paragraphs[start].start():paragraphs[end].end()]
        prose = MATH.sub(" ", text)
        bare = prose.strip(" *#\n\t")
        n = sum(sentence_count(p.group()) for p in paragraphs[start:end+1])
        if n > cap:
            break
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prose)
        if not 1 <= n <= cap or not 15 <= len(words) <= 180:
            continue
        if LABEL.search(text) or re.search(r"\b(?:correct|best|final) answer\b|\bthe answer is\b", text, re.I):
            continue
        if META.search(bare) or bare.endswith(":") or not SUPPORT.search(text):
            continue
        if re.search(r"\b(?:above|previous step|known result|following (?:equation|formula|calculation)|as mentioned|we need to (?:find|determine|calculate))\b", text, re.I):
            continue
        # A whole paragraph must not begin/end inside a math span or a code block.
        if "```" in text or any(text.count(a) != text.count(b) for a,b in [(r"\(",r"\)"),(r"\[",r"\]")]):
            continue
        found.append({"text": text, "start": paragraphs[start].start(), "end": paragraphs[end].end(),
                      "sentences": n, "prose_words": len(words)})
    return found


def extract(response, cap):
    candidates = paragraph_candidates(response, cap)
    return candidates[-1] if candidates else None
