from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SUMO GUI startup outside D2RL.")
    parser.add_argument(
        "--sumo_home",
        default=str(Path(sys.prefix) / "Lib" / "site-packages" / "sumo"),
    )
    parser.add_argument(
        "--sumocfg",
        default="maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg",
    )
    parser.add_argument("--traci", action="store_true")
    args = parser.parse_args()

    sumo_home = Path(args.sumo_home)
    os.environ["SUMO_HOME"] = str(sumo_home)
    os.environ["PATH"] = (
        str(sumo_home / "bin")
        + os.pathsep
        + str(Path(sys.prefix) / "Library" / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    binary = sumo_home / "bin" / ("sumo-gui.exe" if os.name == "nt" else "sumo-gui")
    if not binary.exists():
        binary = Path(shutil.which("sumo-gui") or "sumo-gui")

    print(f"python={sys.executable}")
    print(f"SUMO_HOME={os.environ.get('SUMO_HOME')}")
    print(f"sumo_gui={binary}")
    print(f"sumocfg={Path(args.sumocfg).resolve()}")

    if args.traci:
        import traci

        cmd = [
            str(binary),
            "-c",
            args.sumocfg,
            "--step-length",
            "0.1",
            "--start",
            "--quit-on-end",
        ]
        print("traci.start:", cmd)
        traci.start(cmd, numRetries=10)
        print("connected_time", traci.simulation.getTime())
        traci.simulationStep()
        print("after_step_time", traci.simulation.getTime())
        traci.close()
        return

    cmd = [str(binary), "--version"]
    print("subprocess:", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    print("returncode", proc.returncode)
    print("stdout", proc.stdout)
    print("stderr", proc.stderr)


if __name__ == "__main__":
    main()
