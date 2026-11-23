"""Frozen automatic prose-unit counter; formulas are recorded separately."""
import re
from normalize_full import TEX

def count_units(text):
    """Return auditable automatic prose counts without any inference/download."""
    spans = TEX.findall(text)
    displays = sum(s.startswith(('\\[', '$$')) for s in spans)
    # A math-only block is excluded; an inline equation inside prose stays in
    # that prose unit. No token from a protected math span can create a split.
    s = TEX.sub(' @@MATH@@ ', text)
    s = re.sub(r'(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)', '\n\n', s)
    s = re.sub(r'[*_`#]', '', s)
    dot = '\uE000'
    s = re.sub(r'(?<=\d)\.(?=\d)', dot, s)
    s = re.sub(r'\b(?:e\.g\.|i\.e\.|a\.m\.|p\.m\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|vs\.|etc\.|Fig\.|Eq\.|No\.)',
               lambda m: m.group().replace('.', dot), s, flags=re.I)
    s = re.sub(r'\b(?:[A-Z]\.){2,}', lambda m: m.group().replace('.', dot), s)
    count, fragments = 0, 0
    for block in re.split(r'\n\s*\n', s):
        # Split only after terminal punctuation, allowing closing quotes/brackets.
        pieces = re.split(r'(?<=[.!?])(?=[\s\"\u201d\u2019\)\]]|$)', block)
        for piece in pieces:
            prose = piece.replace('@@MATH@@', '').strip()
            if not re.search(r'[A-Za-z]', prose):
                continue
            count += 1
            if not re.search(r'[.!?][\s\"\u201d\u2019\)\]]*$', prose):
                fragments += 1
    return {'sentence_count_auto': count, 'unpunctuated_units': fragments,
            'display_math_blocks': displays, 'inline_math_spans': len(spans)-displays}
