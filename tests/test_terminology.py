import pytest

from action_semantics.indexed_videos import _parse_inventory_items
from action_semantics.models import ActionTriple, ClipRecord
from action_semantics.month1 import _clip_inventory
from action_semantics.retrieval.lexical import TfidfIndex, production_candidate_text
from action_semantics.retrieval.scorers import StructuredResources, structured_score_for_triples
from action_semantics.terminology import ALIASES, normalize_terminology, normalize_triple


def test_inventory_preserves_grouped_lists_dimensions_and_source():
    raw = 'Safety gear (hard hats, gloves, eye protection), 12" Saw; Drill'
    items = _parse_inventory_items(raw)
    assert [i.name for i in items] == [
        'Safety gear (hard hats, gloves, eye protection)', '12" Saw', 'Drill'
    ]
    assert items[0].raw == 'Safety gear (hard hats, gloves, eye protection)'


def test_inventory_separates_primary_alternatives_purpose_and_placeholders():
    raw = ('Miter Saw Unspecified alternatives: Circular saw (corded, cordless), Handsaw '
           'used for Cutting boards, trim and rails., Water N/A')
    items = _parse_inventory_items(raw)
    assert [i.name for i in items] == ['Miter Saw', 'Water']
    assert items[0].alternatives == ['Circular saw (corded, cordless)', 'Handsaw']
    assert items[0].purpose == 'Cutting boards, trim and rails'
    assert 'Unspecified' in items[0].raw
    assert _parse_inventory_items('Unknown') == []


@pytest.mark.parametrize('query,expected', [
    ('Cut 5/8-inch Gypsum Board', 'Cut 5/8-inch drywall'),
    ('Do not install plasterboard', 'Do not install drywall'),
    ('Use a measuring tape and Allen key', 'Use a tape measure and hex key'),
    ('Cut with a mitre saw', 'Cut with a miter saw'),
    ('Sheetrock joint compound, stucco mud, tap water',
     'Sheetrock joint compound, stucco mud, tap water'),
    ('Pipe wrench and circular saw', 'Pipe wrench and circular saw'),
    ('Smooth, level, dry, cured; remove or install',
     'Smooth, level, dry, cured; remove or install'),
    ('plasterboarding', 'plasterboarding'),
])
def test_aliases_preserve_meaning_and_are_idempotent(query, expected):
    result = normalize_terminology(query)
    assert result == expected
    assert normalize_terminology(result) == result


def test_lexical_alias_query_matches_without_merging_compound_or_changing_input():
    clips = [ClipRecord(clip_id='board', title='drywall'),
             ClipRecord(clip_id='compound', title='Sheetrock joint compound')]
    baseline = TfidfIndex.from_clips(clips)
    challenger = TfidfIndex.from_clips(clips, terminology=True)
    assert baseline.scores('gypsum board')['board'] == 0
    assert challenger.scores('gypsum board')['board'] > 0
    assert challenger.scores('gypsum board')['compound'] == 0
    assert clips[1].title == 'Sheetrock joint compound'


@pytest.mark.parametrize('alias,canonical', ALIASES.items())
def test_every_active_alias_matches_in_both_query_and_candidate_directions(alias, canonical):
    for query, candidate in [(alias, canonical), (canonical, alias)]:
        index = TfidfIndex.from_clips(
            [ClipRecord(clip_id='target', title=candidate),
             ClipRecord(clip_id='distractor', title='unrelated')], terminology=True)
        scores = index.scores(query)
        assert scores == pytest.approx(index.scores(canonical))
        assert scores['target'] > 0
        assert scores['distractor'] == 0


def test_alternatives_are_preserved_but_not_treated_as_present_in_challenger():
    clip = ClipRecord(clip_id='saw', gemini_metadata={'clip': {
        'tools': ['Miter Saw'], 'tool_items': [{
            'name': 'Miter Saw', 'alternatives': ['Circular saw']}],
    }})
    assert 'Circular saw' in production_candidate_text(clip)
    assert 'Circular saw' not in production_candidate_text(clip, include_alternatives=False)
    assert 'circular' not in _clip_inventory(clip, include_alternatives=False)[0]
    assert clip.gemini_metadata['clip']['tool_items'][0]['alternatives'] == ['Circular saw']


def test_structured_alias_alignment_preserves_negation_and_action_direction():
    def triple(record_type, record_id, action, obj, lemmas, negated=False):
        return ActionTriple(record_type=record_type, record_id=record_id,
                            source_field='description', action=action, action_lemma=action,
                            action_text=action, object_text=obj, object_lemmas=lemmas,
                            negated=negated, sentence=f'{action} {obj}',
                            extraction_method='test')
    query = triple('step', 'query', 'install', 'gypsum board', ['board', 'gypsum'])
    target = triple('clip', 'target', 'install', 'drywall', ['drywall'])
    opposite = triple('clip', 'opposite', 'remove', 'drywall', ['drywall'])
    negative = triple('clip', 'negative', 'install', 'drywall', ['drywall'], True)
    resources = StructuredResources(triples=[target, opposite, negative],
                                    verbnet=[], framenet=[], taxonomy=[])
    normalized = normalize_triple(query)
    assert normalized.object_lemmas == ['drywall']
    assert query.object_lemmas == ['board', 'gypsum']
    assert structured_score_for_triples([normalized], 'target', resources)['object_match'] == 1
    for clip_id in ['opposite', 'negative']:
        assert structured_score_for_triples([normalized], clip_id, resources)['structured_score'] == 0
