"""Full 31-action exponential tilt of the frozen NDD in causal lateral phases.

P_new = rho * P_ref + (1-rho) * normalized(P_ref * exp(theta1*a+theta2*a^2)).
All variants have the same action family, regularization, and reference mass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import highd_lane_phase_longitudinal as phase
from .highd_pair_data_inventory import digest, read_json, write_new
from .highd_pair_model_fit import FrozenMarginal


def action_basis(axis):
    return np.column_stack([axis / 2, (axis / 2) ** 2])


def log_probability(x, log_p0, beta, basis, rho):
    logits = log_p0 + (x @ beta) @ basis.T
    tilt = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.logaddexp(np.log(rho) + log_p0, np.log1p(-rho) + tilt), tilt


def objective(beta_flat, x, log_p0, action, basis, rho, l2):
    beta = beta_flat.reshape(x.shape[1], 2)
    logq, log_tilt = log_probability(x, log_p0, beta, basis, rho)
    rows = np.arange(len(x))
    # Responsibility of the tilted component in the observed mixture likelihood.
    responsibility = np.exp(np.log1p(-rho) + log_tilt[rows, action] - logq[rows, action])
    delta = (np.exp(log_tilt) @ basis - basis[action]) * responsibility[:, None]
    grad = x.T @ delta / len(x)
    grad[1:] += l2 * beta[1:]
    loss = -logq[rows, action].mean() + l2 * np.square(beta[1:]).sum() / 2
    return loss, grad.ravel()


def fit(x, data, basis, config, initial=None):
    if initial is None:
        initial = np.zeros((x.shape[1], 2))
    result = minimize(objective, initial.ravel(), args=(x, data["log_p0_full"], data["action_index"], basis,
                      config["reference_mixture_mass"], config["l2"]), jac=True, method="L-BFGS-B",
                      options={k: config[k] for k in ("gtol", "ftol", "maxcor")} | {"maxiter": config["max_iterations"]})
    return result.x.reshape(x.shape[1], 2), {"iterations": int(result.nit), "converged": bool(result.success),
              "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac))), "message": str(result.message)}


def feature_transforms(state, mean, scale):
    """Residualize new blocks against previous inputs using train rows only.

    This prevents duplicate history x phase columns from improving fit merely
    by splitting a shared coefficient across several L2-penalized parameters.
    The baseline columns remain exactly unchanged in every nested model.
    """
    raw = phase.design(state, mean, scale, "interaction")
    transforms = {}
    previous_count = 0
    previous_transform = None
    for kind, count in phase.MODEL_FEATURES.items():
        width = count + 1
        transform = np.eye(width)
        if previous_count:
            base, extra = raw[:, :previous_count], raw[:, previous_count:width]
            projection = np.linalg.lstsq(base, extra, rcond=1e-10)[0]
            residual = extra - base @ projection
            spread = residual.std(axis=0)
            supported = spread > 1e-8
            inverse = np.divide(1, spread, out=np.zeros_like(spread), where=supported)
            transform[:previous_count, :previous_count] = previous_transform
            transform[:previous_count, previous_count:] = -projection * inverse
            transform[previous_count:, previous_count:] = np.diag(inverse)
        transforms[kind] = transform
        previous_count, previous_transform = width, transform
    return transforms


def model_design(state, model, kind):
    raw = phase.design(state, model["mean"], model["scale"], kind)
    transform = model.get("transform_" + kind)
    return raw if transform is None else raw @ transform


def scores(data, model, basis, config):
    rows = np.arange(len(data["action_index"]))
    logs = {"frozen": data["log_p0_full"][rows, data["action_index"]]}
    for kind in phase.MODEL_FEATURES:
        x = model_design(data["state"], model, kind)
        logq, _ = log_probability(x, data["log_p0_full"], model[kind], basis, config["reference_mixture_mass"])
        logs[kind] = logq[rows, data["action_index"]]
    return logs


def run(output, config, split):
    output = Path(output)
    if (output / (split + "_summary.json")).exists():
        raise ValueError("Report exists; use a fresh output")
    contract = read_json(config["freeze_contract"])
    marginal = FrozenMarginal(contract, config["action_thresholds_mps2"])
    basis = action_basis(marginal.axes["acceleration"])
    if not config["full_action_probability"] or not 0 < config["reference_mixture_mass"] < 1:
        raise ValueError("Expected a full-support fine-action reference mixture")
    provenance = {"code_sha256": digest(__file__), "phase_code_sha256": digest(phase.__file__),
                  "clock_code_sha256": digest(Path(__file__).with_name("highd_conditional_lane_start.py")),
                  "domain_sha256": digest(config["domain_config"]), "freeze_contract_sha256": digest(config["freeze_contract"])}
    model_path = output / "lane_phase_fine_v1.npz"
    if split == "train" and model_path.exists():
        raise ValueError("Model exists; use a fresh output")
    if split == "calibration":
        protocol = read_json(output / "train_summary.json")
        if protocol["config"] != config or protocol["provenance"] != provenance or protocol["model_sha256"] != digest(model_path):
            raise ValueError("Fixed model or feature protocol changed")
    output.mkdir(parents=True, exist_ok=True)
    pieces, summaries, events = {"focal": [], "rear": []}, [], []
    for rec in contract[split + "_recordings"]:
        roles, these_events, summary = phase.extract_recording(Path(config["source_root"]), rec, config, marginal)
        summaries.append(summary)
        events.extend(dict(recording=rec, **event) for event in these_events)
        for role in pieces:
            if roles[role] is not None:
                pieces[role].append((rec, roles[role]))
        print(json.dumps(summary), flush=True)
    write_new(output / (split + "_maneuvers.json"), events)
    models, quality, results = {}, {}, {}
    if split == "calibration":
        with np.load(model_path, allow_pickle=False) as artifact:
            for key in artifact.files:
                role, name = key.split("__")
                models.setdefault(role, {})[name] = artifact[key]
    for role, blocks in pieces.items():
        data = {key: np.concatenate([d[key] for _, d in blocks]) for key in blocks[0][1]}
        if split == "train":
            mean = data["state"].astype(float).mean(axis=0)
            scale = data["state"].astype(float).std(axis=0)
            scale[scale < 1e-8] = 1
            model, quality[role] = {"mean": mean, "scale": scale}, {}
            if config.get("orthogonalize_added_features", False):
                model.update({"transform_" + kind: value for kind, value in feature_transforms(data["state"], mean, scale).items()})
            previous = None
            for kind in phase.MODEL_FEATURES:
                x = model_design(data["state"], model, kind)
                initial = np.zeros((x.shape[1], 2))
                if previous is not None:
                    initial[:len(previous)] = previous
                model[kind], quality[role][kind] = fit(x, data, basis, config, initial)
                previous = model[kind]
                print(json.dumps({"role": role, "model": kind, **quality[role][kind]}), flush=True)
            models[role] = model
        model = models[role]
        records = {rec: phase.score_summary(d, scores(d, model, basis, config)) for rec, d in blocks}
        results[role] = {"overall": phase.score_summary(data, scores(data, model, basis, config)),
                         "by_recording": records,
                         "comparisons": {prior + "_to_" + candidate: {
                             "all": phase.cluster_gain(list(records.values()), prior, candidate, config),
                             "two_lane": phase.cluster_gain(list(records.values()), prior, candidate, config, True)}
                             for prior, candidate in (("frozen", "history"), ("history", "phase"), ("phase", "interaction"), ("history", "interaction"))}}
    if split == "train":
        np.savez_compressed(model_path, **{role + "__" + key: value for role, model in models.items() for key, value in model.items()})
    report = {"config": config, "provenance": provenance, "model_sha256": digest(model_path), "split": split,
              "features": phase.FEATURES, "optimization": quality, "recordings": summaries, "roles": results,
              "scope": "Full 31-action conditional role distributions during causally detected lateral motion, including returns and censoring. Separate role models, not residual correlated joint NDD. Calibration is reused development evidence, not confirmatory significance. No validation/test access or runtime changes."}
    write_new(output / (split + "_summary.json"), report)
    print(json.dumps({role: result["overall"] for role, result in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/highd_lane_phase_fine_v1.json"))
    parser.add_argument("--split", choices=("train", "calibration"), default="train")
    args = parser.parse_args()
    run(args.output, read_json(args.config), args.split)


if __name__ == "__main__":
    main()
