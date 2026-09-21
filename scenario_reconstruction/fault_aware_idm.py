"""IDM controller variant that receives auditable scenario fault injections."""
from __future__ import annotations

from mtlsp.controller.vehicle_controller.controller import BaseController
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController

from .cav_fault_model import CAVFaultModel


class FaultAwareIDMController(IDMController):
    def __init__(self, *args, fault_model: CAVFaultModel | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_model = fault_model

    def step(self):
        BaseController.step(self)
        observation = self.vehicle.observation.information
        current_time = self.vehicle.simulator.get_time()
        if self.fault_model is not None and self.fault_model.enabled:
            observation = self.fault_model.transform_observation(current_time, observation)
        self.action, self.mode = self.decision(observation)
        if self.fault_model is not None and self.fault_model.enabled:
            self.action = self.fault_model.delay_control(current_time, self.action)
