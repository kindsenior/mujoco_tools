#!/usr/bin/env -S python3 -i
import gymnasium
import mujoco
import mujoco.viewer
from pathlib import Path
import roslib

package_path = Path(roslib.packages.get_pkg_dir("mujoco_tools"))

# generate model from xml
model = mujoco.MjModel.from_xml_path(str(package_path / "scripts/samples/example.xml"))
# model = mujoco.MjModel.from_xml_path(str(Path(gymnasium.__path__[0]) / "envs" / "mujoco" / "assets" / "humanoid.xml"))

# # generate model from gymnasium
# import mujoco
# env = gymnasium.make("Humanoid-v5")
# env.reset(seed=0)
# model = env.unwrapped.model # access to mujoco's model and data

# data
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    import time
    for _ in range(4000):
        mujoco.mj_step(model, data)
        time.sleep(0.001)
