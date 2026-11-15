"""Supplementary reference numeric-value recall; not an entailment/correctness judge."""
import re


NUMERIC_LABEL_SCHEMA={'version':1,'name':'reference_dotted_numeric_recall',
    'pattern':r'(?<![\w.])v?\d+(?:\.\d+)+(?!\w|\.\w)',
    'proxy':'F1 when no reference values; otherwise .5*F1 + .5*numeric recall',
    'limitation':'presence only; negation, contradiction and unsupported extra claims are not resolved'}


def numeric_values(text):
    return sorted({v.lower().removeprefix('v') for v in re.findall(NUMERIC_LABEL_SCHEMA['pattern'],text,re.I)})


def technical_proxy(reference,answer,f1):
    expected=set(numeric_values(reference));observed=set(numeric_values(answer))
    recall=len(expected&observed)/len(expected) if expected else None
    return {'reference_numeric_values':sorted(expected),'answer_numeric_values':sorted(observed),
        'numeric_recall':recall,'quality_proxy':f1 if recall is None else .5*f1+.5*recall}
