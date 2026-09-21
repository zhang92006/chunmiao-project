from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping

from conf import conf


def run_template(
    template_path: str,
    episode: int,
    experiment_path: str,
    gui: bool = False,
    gui_delay: int = 100,
    epsilon: float | Mapping[str, float] = 0.99,
    proposal_mode: str = "joint_pair",
    online_policy=None,
    frozen_epsilon_source: str = "template",
    simulation_seed: int | None = None,
    online_intervention_budget: int | None = None,
) -> float:
    online_intervention_budget = _validated_online_intervention_budget(
        online_policy, online_intervention_budget
    )
    if online_policy is not None and (proposal_mode != "factorized" or frozen_epsilon_source != "runtime"):
        raise ValueError("Online epsilon requires factorized proposals and runtime epsilon")
    if online_policy is not None and conf.weight_threshold != 0:
        raise ValueError("Online probability audit requires weight_threshold=0")
    if frozen_epsilon_source not in {"template", "runtime"}:
        raise ValueError("Invalid frozen_epsilon_source")
    if simulation_seed is not None:
        import random
        import numpy as np
        random.seed(simulation_seed)
        np.random.seed(simulation_seed)
    valid_modes = {"naturalistic", "factorized", "joint_pair"}
    if proposal_mode not in valid_modes:
        raise ValueError(f"proposal_mode must be one of {sorted(valid_modes)}")
    epsilon = _validated_epsilon(epsilon, proposal_mode)
    conf.experiment_config["mode"] = "behavior_policy"
    conf.simulation_config["epsilon_setting"] = "fixed"
    conf.epsilon_value = epsilon
    conf.simulation_config["gui_flag"] = gui
    conf.simulation_config["gui_delay"] = gui_delay
    conf.simulation_config["gui_start"] = True
    if gui:
        _ensure_sumo_home()
    from mtlsp.simulator import Simulator

    from .environment import ScenarioNADE

    env = ScenarioNADE(template_path, multibv_proposal_mode=proposal_mode)
    env.online_epsilon_policy = online_policy
    env.online_intervention_budget = online_intervention_budget
    env.closed_loop_evaluation = online_policy is not None or simulation_seed is not None
    env.frozen_epsilon_source = frozen_epsilon_source
    if online_policy is not None and env.multi_bv_control_num != 2:
        raise ValueError("Online policy requires a multibv template")
    env.info_extractor.episode_log["scenario_metadata"].update({
        "epsilon_source": "online_policy" if online_policy is not None else "fixed",
        "frozen_epsilon_source": frozen_epsilon_source,
        "simulation_seed": simulation_seed,
        "online_policy": online_policy.metadata if online_policy is not None else None,
        "online_intervention_budget": online_intervention_budget,
    })
    sim = Simulator(
        sumo_net_file_path="./maps/2LaneHighway/2LaneHighway.net.xml",
        sumo_config_file_path="./maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg",
        num_tries=50,
        step_size=0.1,
        action_step_size=0.1,
        lc_duration=1,
        track_cav=gui,
        sublane_flag=True,
        gui_flag=gui,
        output=[],
        experiment_path=experiment_path,
    )
    sim.bind_env(env)
    sim.simulation_seed = simulation_seed
    Path(experiment_path, "crash").mkdir(parents=True, exist_ok=True)
    Path(experiment_path, "tested_and_safe").mkdir(parents=True, exist_ok=True)
    Path(experiment_path, "rejected").mkdir(parents=True, exist_ok=True)
    sim.run(episode)
    return env.info_extractor.weight_result


def _validated_epsilon(
    epsilon: float | Mapping[str, float], proposal_mode: str
) -> float | dict[str, float]:
    if proposal_mode == "naturalistic":
        return 1.0
    if isinstance(epsilon, Mapping):
        if proposal_mode != "factorized":
            raise ValueError("per-BV epsilon is only defined for factorized proposals")
        values = {
            "BV_primary": float(epsilon["BV_primary"]),
            "BV_context": float(epsilon["BV_context"]),
        }
        if not all(0.0 < value < 1.0 for value in values.values()):
            raise ValueError("each per-BV epsilon must lie strictly between zero and one")
        return values
    value = float(epsilon)
    if not 0.0 < value < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    return value


def _validated_online_intervention_budget(online_policy, budget):
    if budget is None:
        return None
    if online_policy is None:
        raise ValueError("online_intervention_budget requires an online policy")
    if isinstance(budget, bool) or int(budget) != budget or int(budget) < 1:
        raise ValueError("online_intervention_budget must be a positive integer")
    return int(budget)


def _ensure_sumo_home() -> None:
    if os.environ.get("SUMO_HOME"):
        _prepend_sumo_bin(Path(os.environ["SUMO_HOME"]))
        return
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "sumo",
        Path(sys.prefix) / "share" / "sumo",
    ]
    for candidate in candidates:
        if (candidate / "tools").is_dir():
            os.environ["SUMO_HOME"] = str(candidate)
            _prepend_sumo_bin(candidate)
            return


def _prepend_sumo_bin(sumo_home: Path) -> None:
    sumo_bin = sumo_home / "bin"
    if not (sumo_bin / "sumo-gui.exe").exists():
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    sumo_bin_text = str(sumo_bin)
    if sumo_bin_text not in path_parts:
        os.environ["PATH"] = sumo_bin_text + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one reconstructed scenario.")
    parser.add_argument("template", help="Path to a scenario template YAML file.")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode id written into the generated JSON.",
    )
    parser.add_argument(
        "--experiment_path",
        default="./data_analysis/raw_data/ScenarioReconstructionDebug",
        help="Directory where NADEInfoExtractor writes episode data.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run the scenario in SUMO GUI.",
    )
    parser.add_argument(
        "--gui_delay",
        type=int,
        default=100,
        help="SUMO GUI delay in milliseconds per simulation step.",
    )
    parser.add_argument(
        "--proposal_mode",
        choices=("naturalistic", "factorized", "joint_pair"),
        default="joint_pair",
        help="Multi-BV action proposal family; naturalistic sets epsilon to 1 exactly.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.99,
        help="Naturalistic-mixture probability for fixed NADE importance sampling.",
    )
    parser.add_argument(
        "--epsilon_primary",
        type=float,
        default=None,
        help="Optional fixed epsilon for BV_primary; requires --epsilon_context.",
    )
    parser.add_argument(
        "--epsilon_context",
        type=float,
        default=None,
        help="Optional fixed epsilon for BV_context; requires --epsilon_primary.",
    )
    args = parser.parse_args()

    if (args.epsilon_primary is None) != (args.epsilon_context is None):
        parser.error("--epsilon_primary and --epsilon_context must be provided together")
    epsilon = args.epsilon
    if args.epsilon_primary is not None:
        epsilon = {
            "BV_primary": args.epsilon_primary,
            "BV_context": args.epsilon_context,
        }

    weight = run_template(
        args.template,
        args.episode,
        args.experiment_path,
        gui=args.gui,
        gui_delay=args.gui_delay,
        epsilon=epsilon,
        proposal_mode=args.proposal_mode,
    )
    print(f"Scenario finished. weight_result={weight}")


if __name__ == "__main__":
    main()
