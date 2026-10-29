"""Candidate feature consumers refuse stale provenance before using cached tensors."""

from models.candidate_contract import verify_provenance, verify_runtime
from models.features import extract_features as _extract, validate_features, fit_ridge_probe as _fit
from models.features import score_ridge_probe as _score
from models.training import runtime_contract


def extract_features(model, panel, *, provenance):
    verify_provenance(provenance)
    return _extract(model, panel, provenance=provenance)


def verify_features(bundle, *, expected_provenance, model):
    verify_provenance(expected_provenance)
    c = bundle['contract']
    verify_runtime(c['runtime'], c['provenance'])
    verify_provenance(c['provenance'])
    from models.diagnostics import state_digest
    if (c['provenance'] != expected_provenance
            or c['runtime'] != runtime_contract(next(model.encoder.parameters()).device.type, model=model)
            or c['encoder_sha256'] != state_digest(model.encoder)
            or c['target_sha256'] != state_digest(model.target_encoder)):
        raise ValueError('Candidate features differ in weights, target, runtime or provenance.')
    validate_features(bundle)


def fit_ridge_probe(bundle, *, expected_provenance, model):
    verify_features(bundle, expected_provenance=expected_provenance, model=model)
    return _fit(bundle)


def score_ridge_probe(probe, bundle, *, expected_provenance, model):
    verify_features(bundle, expected_provenance=expected_provenance, model=model)
    verify_provenance(probe['representation']['provenance'])
    return _score(probe, bundle)
