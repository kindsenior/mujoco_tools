#!/usr/bin/env -S python3 -i
import logging
# logging.basicConfig(level=logging.DEBUG)
import numpy as np
import mujoco
import mujoco.viewer
from mujoco_tools.modeling import sample_robot_biped
from mujoco_tools.visualization import *

def test():
    # create model
    global spec
    spec = mujoco.MjSpec()
    spec.from_string(common_xml)
    sample_robot_biped(spec)

    global model, data
    model = spec.compile()
    data = mujoco.MjData(model)

    # FK
    data.qpos[0:3] = [0,0,0.5] # root pos
    data.qpos[7:] = np.deg2rad([0,0,-30, 60, -30,0, 0,0,-30, 60, -30,0]) # angles
    mujoco.mj_forward(model, data)
    for ee_name in ["right_ee", "left_ee"]:
        print(f"{ee_name} pos (world):", data.site(ee_name).xpos.copy())

    # export to xml
    xml_str = spec.to_xml()
    with open("sample_robot_biped.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)

    # display
    global viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints
    viewer.sync()

if __name__ == "__main__":
    test()