"""Low-speed vehicle execution used only by declared CAV fault-validation runs."""
from __future__ import annotations

from mtlsp.vehicle.vehicle import Vehicle


class FaultAwareVehicle(Vehicle):
    """Apply IDM actions without the project's global 20 m/s lower clamp."""

    def act(self, action):
        self.simulator.set_vehicle_speedmode(self.id, 0)
        self.simulator.set_vehicle_lanechangemode(self.id, 0)
        acceleration = float(action["longitudinal"])
        if action["lateral"] == "central":
            offset = self.simulator.get_vehicle_lateral_lane_position(self.id)
            self.simulator.change_vehicle_sublane_dist(
                self.id, -offset, self.step_size
            )
            self.simulator.change_vehicle_speed(
                self.id, acceleration, self.action_step_size
            )
        else:
            self.simulator.change_vehicle_lane(self.id, action["lateral"], self.lc_duration)
            self.simulator.change_vehicle_speed(self.id, acceleration, self.lc_duration)
