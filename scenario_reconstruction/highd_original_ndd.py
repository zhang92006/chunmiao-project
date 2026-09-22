"""Opt-in alignment of legacy IDM positional geometry to observed bumper gaps.

Only the original longitudinal component receives synthetic front positions.
Real logged observations and original lateral policy inputs remain untouched.
"""
from __future__ import annotations

import copy
import numpy as np


GAP_MODES = ('legacy_position', 'measured_gap')


def longitudinal_observation(obs, reference_length):
    result = copy.deepcopy(obs)
    lead = result.get('Lead')
    if lead is None:
        return result
    gap = float(lead['distance'])
    x = float(result['Ego']['position'][0])
    if not np.isfinite([gap, x, reference_length]).all() or reference_length <= 0:
        raise ValueError('Expected finite observed gap, position and positive reference length')
    position = list(lead['position'])
    position[0] = x + gap + reference_length
    lead['position'] = position
    return result


def original_ndd_pdf(obs, mode='legacy_position'):
    from controller.nddcontroller import NDDController
    import conf.conf as conf

    if mode not in GAP_MODES:
        raise ValueError('Unknown original gap mode')
    if mode == 'legacy_position':
        return NDDController.static_get_ndd_pdf(obs=obs)
    _, longitudinal = NDDController.Longitudinal_NDD(longitudinal_observation(obs, conf.LENGTH))
    _, _, lateral = NDDController.Lateral_NDD(obs)
    total = np.concatenate(([lateral[0], lateral[2]], lateral[1]*np.asarray(longitudinal)))
    return longitudinal, lateral, total


def original_longitudinal_error(records):
    from controller.nddcontroller import NDDController
    import conf.conf as conf

    error = 0.0
    checked = 0
    for row in records:
        logged = np.asarray(row['original_pdf'][2:], dtype=float)
        if logged.sum() == 0:  # Original policy has zero keep-lane probability.
            continue
        _, expected = NDDController.Longitudinal_NDD(
            longitudinal_observation(row['observation'], conf.LENGTH))
        expected = np.asarray(expected, dtype=float)
        if not np.isfinite(expected).all() or expected.sum() <= 0:
            raise ValueError('Invalid reconstructed original longitudinal distribution')
        error = max(error, float(np.max(abs(logged/logged.sum()-expected/expected.sum()))))
        checked += 1
    return {'checked_original_longitudinal_pdfs': checked,
            'maximum_original_longitudinal_reconstruction_error': error}
