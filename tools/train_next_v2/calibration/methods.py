"""Label-free grouping and finite-sample calibration primitives."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
import hashlib
import math

import numpy as np

from tools.train_next_v2.analysis.paired_confirmation import connected_groups


def calibration_split(records, assignments, seed):
    """Split connected calibration groups using only frozen identity metadata."""
    if set(assignments) != {r['song_id'] for r in records}:
        raise ValueError('Original training membership changed')
    domains = defaultdict(list)
    for key, group in connected_groups(records).items():
        roles = {assignments[r['song_id']] for r in group}
        if len(roles) != 1:
            raise ValueError('Connected identity crosses the original B roles')
        if roles != {'calibration'}:
            continue
        names = {r['source_dataset'] for r in group}
        if len(names) != 1:
            raise ValueError('Cross-domain calibration group needs a new protocol')
        domains[next(iter(names))].append((key, group))
    output = []
    for domain, groups in sorted(domains.items()):
        def order(pair):
            return hashlib.sha256(f'{seed}:{domain}:{pair[0]}'.encode()).hexdigest()
        groups = sorted(groups, key=order)
        if len(groups) < 2:
            raise ValueError('Need independent fitting and audit groups')
        for index, (key, group) in enumerate(groups):
            role = 'interval_fit' if index < len(groups) // 2 else 'interval_audit'
            output.extend(dict(song_id=r['song_id'], source_dataset=domain,
                group_id=r['group_id'], connected_group=key, audio_sha256=r['audio_sha256'],
                labels_path=r['labels_path'], role=role) for r in group)
    return sorted(output, key=lambda r: r['song_id'])


def finite_quantile(scores, coverage):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected nonempty finite nonnegative group scores')
    if not 0 < coverage < 1:
        raise ValueError('Coverage must be strictly between zero and one')
    # Decimal avoids ceil(10 * 0.9) being corrupted by binary roundoff.
    rank = int((Decimal(len(values) + 1) * Decimal(str(coverage))).to_integral_value(rounding=ROUND_CEILING))
    return dict(groups=len(values), rank=rank,
                q=float(np.sort(values)[rank - 1]) if rank <= len(values) else None,
                finite=rank <= len(values))


def axis_evidence(member_predictions, target, mask, axis, floor):
    """Return evidence only at this axis' observed native timestamps."""
    p, y, observed = np.asarray(member_predictions), np.asarray(target), np.asarray(mask)
    if (p.ndim != 3 or p.shape[0] != 3 or y.ndim != 2 or y.shape[1] != 2
            or p.shape[1:] != y.shape or observed.shape != y.shape or observed.dtype != bool
            or axis not in (0, 1) or floor <= 0):
        raise ValueError('Three V/A members, separate boolean masks and a positive floor required')
    if not np.isfinite(p).all() or not np.isfinite(y[observed]).all():
        raise ValueError('Non-finite predictions or observed targets')
    valid = observed[:, axis]
    if not valid.any():
        return None
    mean = p.astype(float).mean(axis=0)[valid, axis]
    uncertainty = p.astype(float).std(axis=0, ddof=1)[valid, axis]
    scale = np.maximum(uncertainty, floor)
    error = np.abs(mean - y[valid, axis])
    return dict(error=error, scale=scale, disagreement=uncertainty,
                group_score=float((error / scale).max()), constant_score=float(error.max()))


def fit_intervals(rows, coverage):
    """Group maximum protects all observed timestamps in a connected song group."""
    if not rows or any(r['role'] != 'interval_fit' for r in rows):
        raise ValueError('Only predeclared interval_fit evidence may fit intervals')
    result = {}
    for method, field in [('scaled', 'group_score'), ('constant', 'constant_score')]:
        groups = defaultdict(list)
        for row in rows:
            groups[row['connected_group']].append(row[field])
        result[method] = finite_quantile([max(v) for v in groups.values()], coverage)
    return result


def wilson_interval(successes, total, z=1.959963984540054):
    if not 0 <= successes <= total or total == 0:
        raise ValueError('Invalid binomial counts')
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., center-half), min(1., center+half)]


def audit_intervals(rows, fit, method):
    if not rows or any(r['role'] != 'interval_audit' for r in rows) or method not in ('scaled', 'constant'):
        raise ValueError('Only interval_audit evidence may report audit coverage')
    if not fit['finite'] or fit['q'] is None:
        return dict(finite=False, groups=len({r['connected_group'] for r in rows}))
    group_covered, widths, errors, scales, song_coverages = defaultdict(list), [], [], [], []
    for row in rows:
        scale = row['scale'] if method == 'scaled' else np.ones_like(row['error'])
        width = fit['q'] * scale
        covered = row['error'] <= width + 1e-12
        group_covered[row['connected_group']].append(bool(covered.all()))
        widths.append(width); errors.append(row['error']); scales.append(row['disagreement'])
        song_coverages.append(float(covered.mean()))
    successes = sum(all(v) for v in group_covered.values())
    e, w, u = map(np.concatenate, (errors, widths, scales))
    return dict(finite=True, groups=len(group_covered), covered_groups=successes,
        group_simultaneous_coverage=successes/len(group_covered),
        group_coverage_wilson95=wilson_interval(successes, len(group_covered)),
        point_coverage=float((e <= w + 1e-12).mean()),
        song_macro_point_coverage=float(np.mean(song_coverages)),
        halfwidth_median=float(np.median(w)), halfwidth_p90=float(np.quantile(w, .9)),
        disagreement_mean=float(u.mean()), valid_points=len(e))


def readiness(fit, audit, macro_mae, gates):
    finite = bool(fit['finite'] and audit.get('finite'))
    checks = dict(finite_quantile=finite,
        enough_fitting_groups=fit['groups'] >= gates['minimum_fit_groups'],
        enough_audit_groups=audit['groups'] >= gates['minimum_audit_groups'],
        audit_group_coverage=finite and audit['group_simultaneous_coverage'] >= gates['minimum_audit_group_coverage'],
        error_bound=bool(np.isfinite(macro_mae) and macro_mae <= gates['maximum_song_macro_mae']),
        useful_width=finite and audit['halfwidth_p90'] <= gates['maximum_point_p90_halfwidth'])
    return dict(checks=checks, passes=all(checks.values()))
