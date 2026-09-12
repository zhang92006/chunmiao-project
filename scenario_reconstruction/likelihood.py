"""Fail-closed boundary between scenario diagnostics and legacy D2RL training."""


def require_single_bv_likelihoods(episode):
    metadata = episode.get("scenario_metadata")
    if metadata is not None and metadata.get("likelihood_valid") is not True:
        raise ValueError("Scenario diagnostics have no valid importance likelihood ratio")
    if episode.get("scenario_event_metrics") or any(
        str(key).startswith("forced_") for key in episode.get("weight_step_info", {})
    ):
        raise ValueError("Legacy scripted heuristic weights are not valid likelihood ratios")
    for field in ("weight_step_info", "ndd_step_info", "drl_obs_step_info"):
        if any(isinstance(record, dict) for record in episode.get(field, {}).values()):
            raise ValueError("MultiBV joint records are not supported by the single-BV training environment")
    for field in ("drl_epsilon_step_info", "real_epsilon_step_info"):
        if any(isinstance(record, list) for record in episode.get(field, {}).values()):
            raise ValueError("MultiBV epsilon lists require a joint-policy training implementation")


def has_supported_likelihoods(episode):
    try:
        require_single_bv_likelihoods(episode)
    except ValueError:
        return False
    return True
