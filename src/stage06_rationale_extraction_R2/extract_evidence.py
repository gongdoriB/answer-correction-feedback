"""Bounded extractive compression; no model-written summary or semantic certification."""
import re
from normalize_full import normalize_explanation as normalize_full, TEX
from legacy_excerpt import paragraph_candidates
from sentence_units import count_units

CAP = 5
WORD_CAP = 180
LABEL = re.compile(r'(?i:\b(?:options?|choices?|answer))\s*(?:(?i:is)\s*)?\**\(?[A-J]\)?\**(?=[\s.,:;!?]|$)|(?m:^[ \t]*(?:[-*]\s+)?\**[A-J][.)]\s)')
POLICY = {
    'version':'max5-contiguous-normalized-excerpt-v3', 'sentence_cap':CAP, 'word_cap':WORD_CAP,
    'source':'Same author-released teacher-final-correct GPT-4o outputs as full v2.',
    'normalization':'Reuse full-v2 deterministic label-announcement removal, option-heading normalization and option-reference expansion; preserve math.',
    'selection':'Enumerate contiguous whole normalized-source paragraphs passing original support-cue and dependency filters. Require <=5 legacy sentence/math units AND <=5 automatic prose units counting list fragments; <=180 whitespace words. Select latest eligible ending paragraph, then longest eligible contiguous block. No outcome-based selection.',
    'limits':'Surface filters do not establish that the excerpt retains all key reasoning or supports the correct answer sufficiently. Equations within the selected span are unchanged; omitted context may matter.',
    'purpose':'Provide bounded public rationale excerpts, not a newly generated or human-certified explanation.',
    'input_policy':'No token truncation. All six R2 inputs must fit the existing 4096 input-token cap.',
    'quality':'Teacher final-label agreement and mechanical traceability only; semantic validity/completeness unverified.',
    'reuse':'Original baseline and corresponding R1 reused. New max5 R2 only; never pool with short-v1 or full-v2 R2.',
}


def eligible_candidates(full_text, cap=CAP):
    found=[]
    for candidate in paragraph_candidates(full_text,cap):
        units=count_units(candidate['text'])
        if not 1 <= units['sentence_count_auto'] <= cap:
            continue
        if len(candidate['text'].split()) > WORD_CAP or LABEL.search(TEX.sub(' ',candidate['text'])):
            continue
        found.append({**candidate,**units,'word_count':len(candidate['text'].split())})
    return found


def normalize_explanation(response, options, labels):
    full,reason,edits=normalize_full(response,options,labels)
    if reason:return None,reason,edits
    candidates=eligible_candidates(full)
    if not candidates:
        return None,'no_contiguous_excerpt_passing_max5_surface_filters',edits
    selected=candidates[-1]
    assert full[selected['start']:selected['end']]==selected['text']
    event={k:v for k,v in selected.items() if k!='text'}
    event.update(operation='select_contiguous_max5_excerpt', normalized_full_characters=len(full),
                 omitted_characters=len(full)-len(selected['text']), preserves_all_key_reasoning_verified=False)
    return selected['text'],None,edits+[event]
