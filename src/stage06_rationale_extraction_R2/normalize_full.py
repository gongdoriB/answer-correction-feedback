"""Preserve complete public model explanations, normalizing explicit option labels.

No semantic rewriting, summarization, or logical-quality certification. All edits
are deterministic and logged; full original responses remain in the inputs ZIP.
"""
import re

POLICY = {
    'version': 'full-explanation-label-normalization-v2',
    'sentence_cap': None,
    'word_cap': None,
    'selection': 'All original questions with an author-released GPT-4o final prediction matching gold, mechanically usable explanation, and all R2 inputs <=4096 tokens.',
    'source': 'Author-released GPT-4o (2024-05-13), five-shot predictions; no new API calls.',
    'normalization': ['Remove the standard answer-is-letter announcement, retaining text after it.',
                      'Turn option-list letter markers into bullets, retaining all option text and explanations.',
                      'Resolve residual explicit option-letter references to the exact original option text in quotation marks.',
                      'Remove empty markup/connective-only lines left by answer-announcement removal.'],
    'equations': 'Preserve inline/display TeX and arithmetic; never remove mathematical variable names as option labels.',
    'input_policy': 'No truncation. Exclude an item from all four analysis cells only if any of its six complete R2 inputs exceeds 4096; eligibility is frozen before new inference.',
    'length_comparison': 'Not a causal short-versus-long comparison; this is a separate version, never pooled with the short-excerpt R2 outputs.',
    'quality': 'Gold agreement is a final-label check, not proof that the explanation is valid, complete, or persuasive.',
}
ANNOUNCEMENT = re.compile(r'(?i:\bthe\s+(?:(?:correct|final)\s+)?answer\s+is)\s*\**\(?([A-J])\)?\**(?=[\s.,:;!]|$)[.,:;!]?')
REFERENCE = re.compile(r'(?P<prefix>(?i:\b(?:option|choice))\s+(?:(?i:is)\s+)?)\**\(?(?P<label>[A-J])\)?\**(?=[\s.,:;!?]|$)')
HEADING = re.compile(r'(?m)^(?P<indent>[ \t]*)(?:[-*]\s+)?(?P<bold>\*{0,2})(?P<label>[A-J])(?:[.)]|\):)(?P<close>\*{0,2})\s+')
TEX = re.compile(r'\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$',re.S)
LABEL = re.compile(r'(?i:\b(?:option|choice|answer))\s*(?:(?i:is)\s*)?\**\(?[A-J]\)?\**(?=[\s.,:;!]|$)|(?m:^[ \t]*(?:[-*]\s+)?\**[A-J][.)]\s)')


def normalize_explanation(response, options, labels):
    """Return text + exact edit log, or exclusion reason; no data-driven length cap."""
    edits=[]
    def recorded(kind, match, replacement):
        edits.append({'operation':kind,'before':match.group(),'after':replacement})
        return replacement
    # Protect mathematics from all prose-only label normalization.
    math=[]
    def protect(match):
        math.append(match.group());return f'@@MATHSPAN{len(math)-1}@@'
    text=TEX.sub(protect,response)
    text=ANNOUNCEMENT.sub(lambda m:recorded('remove_letter_announcement',m,''),text)
    def heading(m):
        # Retain content and balanced bolding; labels on complete option list lines only.
        if m['label'] not in labels:return m.group()
        bold=m['bold'] if not m['close'] else ''
        return recorded('option_heading_to_bullet',m,m['indent']+'- '+bold)
    text=HEADING.sub(heading,text)
    def reference(m):
        label=m['label']
        if label not in labels:return m.group()
        exact=options[labels.index(label)]
        # This preserves the referent; it is not a newly authored reason.
        return recorded('resolve_option_reference',m,m['prefix']+'“'+exact+'”')
    text=REFERENCE.sub(reference,text)
    kept=[]
    for line in text.splitlines():
        if re.fullmatch(r'\s*(?:\*+)?\s*(?:(?i:thus|therefore|hence|so)[,:.]?)?\s*(?:\*+)?\s*',line):
            if line.strip():edits.append({'operation':'remove_empty_line_after_normalization','before':line,'after':''})
            kept.append('')
        else:kept.append(line)
    text='\n'.join(kept)
    for i,span in enumerate(math):text=text.replace(f'@@MATHSPAN{i}@@',span)
    text=re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',text).strip()
    if LABEL.search(TEX.sub(' ',text)):
        return None,'unresolved_explicit_option_label',edits
    if len(re.findall(r'[A-Za-z]+',TEX.sub(' ',text))) < 10:
        return None,'no_substantive_prose_after_normalization',edits
    if any(text.count(a)!=text.count(b) for a,b in [(r'\(',r'\)'),(r'\[',r'\]')]):
        return None,'unbalanced_math_delimiters',edits
    return text,None,edits
