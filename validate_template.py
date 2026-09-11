from __future__ import annotations

import argparse

from scenario_reconstruction.templates import load_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a scenario template.")
    parser.add_argument("template", help="Path to a scenario template YAML file.")
    args = parser.parse_args()

    template = load_template(args.template)
    print(f"Template OK: {template.template_id}")
    print(f"Vehicles: 1 CAV + {len(template.actors)} BV")
    print(f"Events: {len(template.events)}")
    print(f"Perturbations: {len(template.perturbations)}")


if __name__ == "__main__":
    main()
