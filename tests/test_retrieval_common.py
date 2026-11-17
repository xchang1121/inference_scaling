"""Shared invariants retained from the retired stepwise controller tests."""
import json

from inference_scaling.retrieval.common import parse_plan, parse_support, step_char_boundaries


def test_boundaries_ignore_versions_lists_and_code():
    text = '1. Version 1.2, e.g. v2.3, is selected. Next'
    assert step_char_boundaries(text) == [text.index('selected.')+len('selected.')]
    assert step_char_boundaries('```\nfoo.\n\nbar!\n```') == []
    assert step_char_boundaries('Use `x. y!` safely') == []
    assert step_char_boundaries('A line\ncontinues') == []
    assert step_char_boundaries('A complete paragraph\n\nNext')


def test_plan_schema_strict_boolean_and_no_markdown_retry():
    for text in ('{"need":"false"}', '```json\n{"need":false}\n```', '{"need":true}', 'null'):
        assert parse_plan(text)[0] is None
    assert parse_plan('{"need":false}') == ({'need': False}, None)
    assert parse_plan('{"need":true,"fact":"timeout","query":"Mercury timeout"}')[0]['need']


def test_support_requires_verbatim_exposed_body():
    packet = {'bodies': [{'text': 'Mercury uses a timeout of twenty seven seconds.'}]}
    valid = {'use': True, 'index': 0, 'quote': 'uses a timeout of twenty seven seconds'}
    assert parse_support(json.dumps(valid), packet)[0]['use']
    for update in ({'index': True}, {'quote': 'uses a timeout of forty seconds'},
                   {'use': 'true'}, {'quote': 'Mercury'}):
        assert parse_support(json.dumps({**valid, **update}), packet)[0] is None
