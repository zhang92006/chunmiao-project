import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from controller.nadeglobalcontroller import NADEBVGlobalController
from scenario_reconstruction.shrp2_freeze_collision_proposal import (
    freeze_collision_proposals,
)


class FrozenCollisionProposalTests(unittest.TestCase):
    def test_freeze_keeps_only_sources_with_target_collision(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps({
                "template_id": "source",
                "description": "source",
                "map": "2Lane",
                "route": "route_0",
                "duration": 12.0,
                "tags": ["multibv", "train"],
                "ego": {"id": "CAV"},
                "actors": [],
                "events": [],
                "perturbations": [],
                "bridge_metadata": {"source_category": "NearCrash"},
            }), encoding="utf-8")
            action_distribution = [
                {"value": 2, "probability": 0.8},
                {"value": 22, "probability": 0.2},
            ]
            summary = root / "cem.json"
            summary.write_text(json.dumps({"source_results": [
                {
                    "source_event_id": 1,
                    "source_template": str(source),
                    "best_candidate": {
                        "target_collision_count": 1,
                        "parameters": {
                            "start_time_s": 1.0,
                            "context_delay_s": 0.5,
                            "primary_duration_s": 1.5,
                            "context_duration_s": 1.0,
                        },
                    },
                    "generations": [{"updated_distributions": {
                        "primary_action_id": action_distribution,
                        "context_action_id": action_distribution,
                    }}],
                },
                {
                    "source_event_id": 2,
                    "source_template": str(source),
                    "best_candidate": {"target_collision_count": 0},
                    "generations": [],
                },
            ]}), encoding="utf-8")

            result = freeze_collision_proposals(summary, root / "frozen", epsilon=0.05)

            self.assertEqual(result["frozen_proposal_count"], 1)
            self.assertEqual(result["excluded_count"], 1)
            template = json.loads(
                Path(result["records"][0]["template_path"]).read_text(encoding="utf-8")
            )
            proposal = template["bridge_metadata"]["frozen_collision_proposal"]
            self.assertEqual(len(proposal["agents"]["BV_primary"]["action_pdf"]), 33)
            self.assertAlmostEqual(
                sum(proposal["agents"]["BV_primary"]["action_pdf"]), 1.0
            )
            self.assertEqual(template["events"], [])

    def test_controller_uses_frozen_pdf_only_inside_declared_window(self):
        controller = NADEBVGlobalController.__new__(NADEBVGlobalController)
        controller.control_log = {}
        controller.env = SimpleNamespace(
            multibv_proposal_mode="factorized",
            simulator=SimpleNamespace(get_time=lambda: 1.2),
            scenario_template=SimpleNamespace(bridge_metadata={
                "frozen_collision_proposal": {
                    "proposal_id": "p1",
                    "agents": {
                        "BV_primary": {
                            "start_time_s": 1.0,
                            "duration_s": 1.0,
                            "epsilon": 0.05,
                            "action_pdf": [1.0] + [0.0] * 32,
                        },
                        "BV_context": {
                            "start_time_s": 2.0,
                            "duration_s": 1.0,
                            "epsilon": 0.05,
                            "action_pdf": [0.0, 1.0] + [0.0] * 31,
                        },
                    },
                }
            }),
        )
        bvs = [
            SimpleNamespace(
                id="BV_primary",
                controller=SimpleNamespace(get_NDD_possi=lambda: np.full(33, 1 / 33)),
            ),
            SimpleNamespace(
                id="BV_context",
                controller=SimpleNamespace(get_NDD_possi=lambda: np.full(33, 1 / 33)),
            ),
        ]

        proposal = controller._frozen_collision_proposal(
            bvs, [np.zeros(33), np.zeros(33)]
        )

        self.assertTrue(proposal["active"])
        self.assertEqual(proposal["debug"]["active_bv_ids"], ["BV_primary"])
        self.assertEqual(proposal["epsilon_by_bv_id"], {
            "BV_primary": 0.05,
            "BV_context": 1.0,
        })
        self.assertAlmostEqual(proposal["criticality_arrays"][0][0], 1.0)

    def test_controller_masks_frozen_actions_without_naturalistic_support(self):
        controller = NADEBVGlobalController.__new__(NADEBVGlobalController)
        controller.control_log = {}
        controller.env = SimpleNamespace(
            multibv_proposal_mode="factorized",
            simulator=SimpleNamespace(get_time=lambda: 1.2),
            scenario_template=SimpleNamespace(bridge_metadata={
                "frozen_collision_proposal": {
                    "proposal_id": "p1",
                    "agents": {
                        "BV_primary": {
                            "start_time_s": 1.0,
                            "duration_s": 1.0,
                            "epsilon": 0.05,
                            "action_pdf": [0.75, 0.25] + [0.0] * 31,
                        },
                    },
                }
            }),
        )
        naturalistic = np.asarray([0.0, 0.4, 0.6] + [0.0] * 30)
        bv = SimpleNamespace(
            id="BV_primary",
            controller=SimpleNamespace(get_NDD_possi=lambda: naturalistic),
        )

        proposal = controller._frozen_collision_proposal([bv], [np.zeros(33)])

        self.assertEqual(proposal["criticality_arrays"][0][0], 0.0)
        self.assertEqual(proposal["criticality_arrays"][0][1], 1.0)
        self.assertEqual(
            proposal["debug"]["removed_unsupported_mass_by_bv_id"],
            {"BV_primary": 0.75},
        )


if __name__ == "__main__":
    unittest.main()
