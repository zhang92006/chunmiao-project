"""Physical check of the published command adapter on the installed SUMO."""
from pathlib import Path
import os
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('acceleration,duration',[(2.,.1),(-4.,.1),(2.,1.),(-4.,1.)])
def test_requested_acceleration_reaches_vehicle(acceleration,duration):
    import sumo
    os.environ['SUMO_HOME'] = sumo.SUMO_HOME
    import libsumo as traci
    from mtlsp.simulator import Simulator
    root=Path(__file__).resolve().parents[1]
    traci.start([str(Path(sumo.SUMO_HOME)/'bin/sumo'),'-n',str(root/'maps/2LaneHighway/2LaneHighway.net.xml'),
        '--step-length','0.1','--seed','17','--no-step-log','true','--no-warnings','true'])
    try:
        traci.route.add('probe',['0to1'])
        traci.vehicle.add('probe','probe',departSpeed='30',departPos='100')
        traci.simulationStep()
        traci.vehicle.setSpeedMode('probe',0)
        traci.vehicle.setLaneChangeMode('probe',0)
        before=traci.vehicle.getSpeed('probe')
        Simulator.change_vehicle_speed(SimpleNamespace(step_size=.1),'probe',acceleration,duration)
        previous=before
        for _ in range(round(duration/.1)):
            traci.simulationStep()
            current=traci.vehicle.getSpeed('probe')
            assert (current-previous)/.1 == pytest.approx(acceleration,abs=1e-7)
            previous=current
        assert previous-before == pytest.approx(acceleration*duration,abs=1e-7)
    finally:
        traci.close()
