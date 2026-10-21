import copy

import pytest
import torch

from models.evaluation import model_identity
from models.features import (FEATURE_RECIPE, extract_features, feature_digest, fit_ridge_probe,
                             raw_feature_diagnostics, score_ridge_probe)
from models.training import rng_state
from test_training import assert_tree_equal
from test_vjepa2 import model, pixels


def bundle(panel='train', *, prefix='v', groups='g', x=None, labels=(3, 3, 17, 17)):
    x = torch.tensor([[1., 3., 5.], [2., 3., 2.], [4., 3., 1.], [5., 3., -2.]]) if x is None else x
    return {'features': x, 'rows': [dict(video_id=f'{prefix}{i}', source_group_id=f'{groups}{i}', class_id=str(c))
                                   for i, c in enumerate(labels)],
            'contract': dict(recipe=FEATURE_RECIPE, panel=panel, panel_identity='fixture',
                             encoder_sha256='context', target_sha256='target', runtime='cpu-fixture', provenance={})}


def test_dual_ridge_matches_independent_primal_solution_and_unpenalized_intercept():
    training = bundle()
    probe = fit_ridge_probe(training)
    x = training['features'].double()
    scale = x.std(0, correction=0)
    scale[scale == 0] = 1.
    design = torch.cat(((x - x.mean(0)) / scale, torch.ones((4, 1), dtype=torch.float64)), dim=1)
    y = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=torch.float64)
    penalty = torch.diag(torch.tensor([1., 1., 1., 0.], dtype=torch.float64))
    expected = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    torch.testing.assert_close(probe['weights'], expected[:-1], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(probe['intercept'], expected[-1], rtol=1e-12, atol=1e-12)
    assert probe['classes'].tolist() == [3, 17]
    assert probe['scale'][1] == 1 and probe['weights'][1].abs().max() == 0
    assert score_ridge_probe(probe, training)['predictions'] == [3, 3, 17, 17]


def test_heldout_scoring_uses_training_statistics_without_refitting():
    training = bundle()
    probe = fit_ridge_probe(training)
    before = copy.deepcopy(probe)
    validation = bundle('exploration_validation', prefix='new', groups='heldout', x=training['features'] + 100)
    result = score_ridge_probe(probe, validation)
    expected = ((validation['features'].double() - probe['mean']) / probe['scale']) @ probe['weights'] + probe['intercept']
    assert result['predictions'] == probe['classes'][expected.argmax(1)].tolist()
    assert_tree_equal(before, probe)
    assert not probe['validation_used_for_fit']


@pytest.mark.parametrize('panel', ['train_monitor', 'exploration_validation', 'confirmation_validation', 'test'])
def test_probe_fit_rejects_every_nontraining_panel(panel):
    with pytest.raises(ValueError, match='training'):
        fit_ridge_probe(bundle(panel))


@pytest.mark.parametrize('defect', ['group_overlap', 'video_overlap', 'new_class', 'changed_encoder',
                                  'changed_source', 'changed_runtime', 'test'])
def test_scoring_rejects_leakage_and_incompatible_representations(defect):
    probe = fit_ridge_probe(bundle())
    evaluation = bundle('exploration_validation', prefix='new', groups='heldout')
    if defect == 'group_overlap':
        evaluation['rows'][0]['source_group_id'] = 'g0'
    elif defect == 'video_overlap':
        evaluation['rows'][0]['video_id'] = 'v0'
    elif defect == 'new_class':
        evaluation['rows'][0]['class_id'] = '9'
    elif defect == 'changed_encoder':
        evaluation['contract']['encoder_sha256'] = 'updated'
    elif defect == 'changed_source':
        evaluation['contract']['provenance'] = {'different': True}
    elif defect == 'changed_runtime':
        evaluation['contract']['runtime'] = 'other'
    else:
        evaluation['contract']['panel'] = 'test'
    with pytest.raises(ValueError):
        score_ridge_probe(probe, evaluation)


def test_raw_variance_and_rank_measurements_precede_standardization():
    constant = raw_feature_diagnostics(torch.ones((4, 8)))
    assert constant['population_variance_mean'] == 0
    assert constant['numerical_centered_rank'] == constant['covariance_entropy_effective_rank'] == 0
    rank_one = torch.tensor([[1., 2., 4.], [2., 4., 8.], [4., 8., 16.]], dtype=torch.float64)
    diagnostics = raw_feature_diagnostics(rank_one)
    assert diagnostics['numerical_centered_rank'] == 1
    assert diagnostics['covariance_entropy_effective_rank'] == pytest.approx(1.)
    assert diagnostics['rank_upper_bound'] == 2
    torch.testing.assert_close(torch.tensor(diagnostics['population_variance_mean'], dtype=torch.float64),
                               rank_one.var(0, correction=0).mean())
    assert diagnostics['computed_before_standardization']
    isotropic = raw_feature_diagnostics(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))
    assert isotropic['covariance_entropy_effective_rank'] == pytest.approx(2.)


def test_full_token_features_match_upstream_and_preserve_state(model, pixels):
    panel = type('Panel', (), {'panel': 'train', 'identity': {'data': 'fixture'},
        'rows': bundle(labels=(3, 17))['rows'], 'pixels': lambda self, i: pixels[i:i+1]})()
    model.train()
    rng, before = rng_state('cpu'), model_identity(model)
    extracted = extract_features(model, panel, provenance={'source': 'fixture'})
    assert_tree_equal(rng, rng_state('cpu'))
    assert model.training and model.encoder.training and not model.target_encoder.training
    assert model_identity(model) == before and all(p.grad is None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        expected = torch.cat([model.encoder(model.normalize(pixels[i:i+1])).mean(1) for i in range(2)])
    assert torch.equal(extracted['features'], expected)
    digest = feature_digest(extracted)
    extracted['seconds'] += 10  # Timings cannot change scientific feature identity.
    assert feature_digest(extracted) == digest
    extracted['features'][0, 0] += 1
    assert feature_digest(extracted) != digest


def test_feature_extraction_failure_restores_mode_and_rng(model, pixels):
    def fail(self, i):
        torch.rand(10)
        raise RuntimeError('interrupted')
    panel = type('Panel', (), {'panel': 'train', 'rows': bundle()['rows'], 'pixels': fail})()
    model.train()
    before = rng_state('cpu')
    with pytest.raises(RuntimeError, match='interrupted'):
        extract_features(model, panel, provenance={})
    assert_tree_equal(before, rng_state('cpu'))
    assert model.training and not model.target_encoder.training
