"""Synthetic judgments verify handoff behavior; none are research observations."""

import csv
import json

import pytest

from action_semantics.io_utils import sha256_file, write_csv
from action_semantics.review_analysis import analyze_review, compare_reviewers


def _load_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def _packet(tmp_path, answers=('a', 'b', 'tie', 'unjudgeable')):
    kit = tmp_path / 'kit'
    (kit / 'private').mkdir(parents=True)
    rows = [{'review_id': f'r{i}', 'step_id': f's{i}', 'query': 'Remove faucet',
             'a_clip_id': 'remove', 'b_clip_id': 'install', 'preference': answer,
             'confidence': '', 'notes': ''} for i, answer in enumerate(answers)]
    ext = [{'review_id': 'e1', 'review_group': 'complete_direct', 'sentence': 'Remove faucet',
            'action_correct': 'yes', 'object_correct': 'no', 'tool_correct': 'unjudgeable', 'notes': ''}]
    write_csv(kit / 'preference_review.csv', rows)
    write_csv(kit / 'extraction_review.csv', ext)
    key = {'rows': {r['review_id']: {
        'immutable': {k: v for k, v in r.items() if k not in ('preference', 'confidence', 'notes')},
        'stratum': 'structured_zero_evidence',
        'frozen_scores': {'lexical': {'a': 1.0, 'b': 0.0}, 'structured': {'a': 0.0, 'b': 0.0},
                          'hybrid': {'a': 0.5, 'b': 0.0}}} for r in rows},
        'steps': {r['step_id']: {'id': r['step_id'], 'title': 'Remove faucet', 'description': '',
                               'tools': [], 'materials': []} for r in rows},
        'extraction_rows': {'e1': {k: v for k, v in ext[0].items() if k not in
                                 ('action_correct', 'object_correct', 'tool_correct', 'notes')}}}
    (kit / 'private/review_key.json').write_text(json.dumps(key))
    (kit / 'manifest.json').write_text(json.dumps({'schema_version': 2,
        'review_key_sha256': sha256_file(kit / 'private/review_key.json')}))
    return dict(kit=kit, extraction=kit / 'extraction_review.csv', preferences=kit / 'preference_review.csv',
                output_dir=tmp_path / 'analysis', judge='synthetic-test-only', draws=30)


def test_analysis_keeps_ab_ties_abstentions_and_paired_differences(tmp_path):
    args = _packet(tmp_path)
    result = analyze_review(**args)
    assert result['status'] == 'complete'
    assert result['preferences']['binary_judgments'] == 2
    assert result['preferences']['methods']['lexical']['mean'] == 0.5
    assert result['preferences']['methods']['structured']['mean'] == 0.5
    assert result['preferences']['paired_deltas_vs_lexical']['hybrid']['mean'] == 0
    assert result['extraction']['overall_descriptive']['tool_correct']['precision'] is None
    assert (args['output_dir'] / 'observed-pairs/pairwise.jsonl').exists()
    manifest = json.loads((args['output_dir'] / 'manifest.json').read_text())
    for path, digest in manifest['outputs'].items():
        assert sha256_file(args['output_dir'] / path) == digest
    assert '95\\%' in (args['output_dir'] / 'review-results.tex').read_text()


def test_analysis_refuses_pending_without_partial_flag(tmp_path):
    args = _packet(tmp_path, ('',))
    with pytest.raises(ValueError, match='pending'):
        analyze_review(**args)
    assert not args['output_dir'].exists()
    result = analyze_review(**args, allow_partial=True)
    assert result['status'] == 'partial'
    assert result['preferences']['methods']['lexical']['mean'] is None
    assert not (args['output_dir'] / 'observed-pairs').exists()
    assert 'incomplete' in (args['output_dir'] / 'review-results.tex').read_text()


def test_all_abstentions_complete_without_inventing_binary_results(tmp_path):
    args = _packet(tmp_path, ('tie', 'unjudgeable'))
    result = analyze_review(**args)
    assert result['status'] == 'complete'
    assert result['preferences']['methods']['lexical']['mean'] is None
    excluded = (args['output_dir'] / 'excluded_judgments.jsonl').read_text().splitlines()
    assert len(excluded) == 2


@pytest.mark.parametrize('change', ['sentence', 'missing', 'duplicate', 'label', 'key', 'column'])
def test_analysis_rejects_corrupt_extraction_or_key(tmp_path, change):
    args = _packet(tmp_path)
    rows = _load_rows(args['extraction'])
    if change == 'sentence':
        rows[0]['sentence'] = 'Install faucet'
    elif change == 'missing':
        rows[0]['review_id'] = 'unknown'
    elif change == 'duplicate':
        rows.append(dict(rows[0]))
    elif change == 'label':
        rows[0]['action_correct'] = 'probably'
    elif change == 'column':
        del rows[0]['action_correct']
    else:
        (args['kit'] / 'private/review_key.json').write_text('{}')
    write_csv(args['extraction'], rows)
    with pytest.raises(ValueError):
        analyze_review(**args)
    assert not args['output_dir'].exists()


def test_reviewer_comparison_excludes_abstentions_and_leaves_adjudication_blank(tmp_path):
    args = _packet(tmp_path, ('a', 'b', 'tie', 'unjudgeable'))
    other = tmp_path / 'second.csv'
    rows = _load_rows(args['preferences'])
    rows[1]['preference'] = 'a'
    rows[-1]['preference'] = ''
    write_csv(other, rows)
    result = compare_reviewers(kit=args['kit'], first=args['preferences'], second=other,
                               output_dir=args['output_dir'])
    assert result['jointly_judged'] == 3
    assert result['raw_agreement'] == pytest.approx(2 / 3)
    assert result['cohen_kappa'] == pytest.approx(0.5)
    assert result['different_responses'] == 2
    assert all(r['adjudicated_preference'] == '' for r in _load_rows(args['output_dir'] / 'disagreements.csv'))


def test_reordered_sheets_and_bom_are_accepted_but_existing_results_are_preserved(tmp_path):
    args = _packet(tmp_path)
    rows = _load_rows(args['preferences'])
    write_csv(args['preferences'], reversed(rows))
    args['preferences'].write_text('\ufeff' + args['preferences'].read_text())
    result = analyze_review(**args)
    assert result['preferences']['binary_judgments'] == 2
    with pytest.raises(FileExistsError):
        analyze_review(**args)


def test_kappa_undefined_when_everyone_chooses_same_category(tmp_path):
    args = _packet(tmp_path, ('a', 'a'))
    result = compare_reviewers(kit=args['kit'], first=args['preferences'], second=args['preferences'],
                               output_dir=args['output_dir'])
    assert result['raw_agreement'] == 1
    assert result['cohen_kappa'] is None
    assert _load_rows(args['output_dir'] / 'disagreements.csv') == []


@pytest.mark.parametrize('change', ['partial', 'modified_tex'])
def test_manuscript_refuses_incomplete_or_modified_review_results(tmp_path, change):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location('paper_tables', Path(__file__).parents[1] / 'paper/generate_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = _packet(tmp_path, ('',) if change == 'partial' else ('a',))
    analyze_review(**args, allow_partial=True)
    if change == 'modified_tex':
        (args['output_dir'] / 'review-results.tex').write_text('altered results')
    with pytest.raises(ValueError, match='completed diagnostic review|hash changed'):
        module.generate(tmp_path / 'unused-suite', tmp_path / 'generated/results.tex', args['output_dir'])
    assert not (tmp_path / 'generated/results.tex').exists()


def test_manuscript_rejects_review_from_another_experiment_suite(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location('paper_tables', Path(__file__).parents[1] / 'paper/generate_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = _packet(tmp_path, ('a',))
    analyze_review(**args)
    suite = tmp_path / 'different-suite'
    suite.mkdir()
    (suite / 'summary.json').write_text('{"status": "complete"}')
    (suite / 'manifest.json').write_text(json.dumps({'outputs': [
        {'path': 'summary.json', 'sha256': sha256_file(suite / 'summary.json')}]}))
    with pytest.raises(ValueError, match='different experiment suites'):
        module.generate(suite, tmp_path / 'generated/results.tex', args['output_dir'])
