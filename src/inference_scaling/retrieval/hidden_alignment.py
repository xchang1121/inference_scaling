"""Supervised frozen-hidden to frozen-document-space alignment, trained separately.

No raw hidden/document dot product is treated as semantic similarity. The learned
map and document index must share a checked artifact provenance.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from inference_scaling.retrieval.linear_probe import fingerprint
from inference_scaling.retrieval.probe_features import pooled_feature_schema, validate_feature_schema


def fit_alignment(features, targets, *, alpha=10., input_rank=128, provenance=None,
                  pool=32, capture_layer=None):
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 128 or y.shape != (len(x), 768) or len(x) < 2:
        raise ValueError('alignment requires paired128-dimensional hidden and768-dimensional document vectors')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid alignment training data or regularization')
    if type(input_rank) is not int or not 1 <= input_rank <= 128:
        raise ValueError('input rank must be an integer in1..128')
    schema = pooled_feature_schema(pool, capture_layer=capture_layer)
    norms = np.linalg.norm(y, axis=1)
    if (norms < 1e-12).any():
        raise ValueError('zero target vector')
    y = y/norms[:, None]
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    z, intercept = (x-center)/scale, y.mean(0)
    if input_rank == 128:
        weights = np.linalg.solve(z.T@z+alpha*np.eye(128), z.T@(y-intercept))
    else:
        # This function sees only its training fold. Fold scaling and PCA must
        # never be fitted on validation examples, even without their labels.
        _, _, right = np.linalg.svd(z, full_matrices=False)
        effective_rank = min(input_rank, len(x)-1)
        basis = right[:effective_rank].T
        reduced = z@basis
        weights = basis@np.linalg.solve(reduced.T@reduced+alpha*np.eye(effective_rank),
                                       reduced.T@(y-intercept))
    model = {'version': 1, 'kind': 'hidden_document_ridge', 'feature_schema': schema,
        'input_dimensions': 128, 'output_dimensions': 768, 'alpha': alpha,
        'center': center.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(),
        'intercept': intercept.tolist(), 'training_rows': len(x), 'provenance': provenance or {},
        'target_scope': 'probe_train source-document centroid in frozen E5 space; not answer quality'}
    if input_rank != 128:
        model['input_pca'] = {'requested_rank': input_rank, 'effective_rank': effective_rank,
                             'fit_scope': 'training rows after training-only normalization',
                             'deployment': 'basis multiplied into128x768 weights'}
    model['sha256'] = fingerprint(model)
    return model


def project_hidden(model, hidden):
    validate_alignment_model(model)
    x = np.asarray(hidden, dtype=np.float64)
    center, scale, weights, intercept = (np.asarray(model[k], dtype=np.float64)
                                        for k in ('center', 'scale', 'weights', 'intercept'))
    if (x.shape != (128,) or center.shape != x.shape or scale.shape != x.shape
            or weights.shape != (128, 768) or intercept.shape != (768,)
            or not all(np.isfinite(a).all() for a in (x, center, scale, weights, intercept))
            or (scale <= 0).any()):
        raise ValueError('invalid alignment dimensions or parameters')
    vector = ((x-center)/scale)@weights+intercept
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('degenerate projected query')
    return vector/norm


def validate_alignment_model(model):
    if model['sha256'] != fingerprint(model):
        raise ValueError('alignment hash mismatch')
    if (model.get('version') != 1 or model.get('kind') != 'hidden_document_ridge'
            or model.get('input_dimensions') != 128 or model.get('output_dimensions') != 768):
        raise ValueError('unsupported alignment model schema')
    validate_feature_schema(model['feature_schema'])


class DocumentVectors:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory/'complete.json').read_text())
        for name, expected in self.manifest['files'].items():
            if Path(name).name != name or hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('document vector artifact hash mismatch')
        self.ids = json.loads((directory/'ids.json').read_text())
        self.vectors = np.load(directory/'vectors.npy', mmap_mode='r', allow_pickle=False)
        if (len(set(self.ids)) != len(self.ids) or self.vectors.shape != (len(self.ids), 768)
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-5)):
            raise ValueError('invalid document vector index')
        self.positions = {name: i for i, name in enumerate(self.ids)}
        self.sha256 = self.manifest['files']['vectors.npy']

    def search(self, vector, *, k=64, exclude=frozenset()):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (768,) or not np.isfinite(vector).all() or k < 1:
            raise ValueError('invalid document-vector query')
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            raise ValueError('zero query vector')
        scores = self.vectors@(vector/norm)
        # IDs are the explicit tie breaker, independent of input order.
        ranked = sorted((i for i, name in enumerate(self.ids) if name not in exclude),
                        key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in ranked]

    def centroid(self, document_ids):
        if not document_ids or any(name not in self.positions for name in document_ids):
            raise ValueError('source document absent from frozen index')
        mean = np.mean(self.vectors[[self.positions[name] for name in document_ids]], axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-12:
            raise ValueError('degenerate source centroid')
        return mean/norm
