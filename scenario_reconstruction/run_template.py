from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from conf import conf


def run_template(
    template_path: str,
    episode: int,
    experiment_path: str,
    gui: bool = False,
    gui_delay: int = 100,
    epsilon: float = 0.99,
) -> float:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    conf.experiment_config["mode"] = "behavior_policy"
    conf.simulation_config["epsilon_setting"] = "fixed"
    conf.epsilon_value = float(epsilon)
    conf.simulation_config["gui_flag"] = gui
    conf.simulation_config["gui_delay"] = gui_delay
    conf.simulation_config["gui_start"] = True
    if gui:
        _ensure_sumo_home()
    from mtlsp.simulator import Simulator

    from .environment import ScenarioNADE

    env = ScenarioNADE(template_path)
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
    Path(experiment_path, "crash").mkdir(parents=True, exist_ok=True)
    Path(experiment_path, "tested_and_safe").mkdir(parents=True, exist_ok=True)
    Path(experiment_path, "rejected").mkdir(parents=True, exist_ok=True)
    sim.run(episode)
    return env.info_extractor.weight_result


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
        "--epsilon",
        type=float,
        default=0.99,
        help="Naturalistic-mixture probability for fixed NADE importance sampling.",
    )
    args = parser.parse_args()

    weight = run_template(
        args.template,
        args.episode,
        args.experiment_path,
        gui=args.gui,
        gui_delay=args.gui_delay,
        epsilon=args.epsilon,
    )
    print(f"Scenario finished. weight_result={weight}")


if __name__ == "__main__":
    main()
