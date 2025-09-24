import mujoco
import numpy as np

common_xml = r"""
<mujoco model="scene">
  <asset>
    <!-- background -->
    <texture name="whitegrad" type="skybox" builtin="gradient"
             rgb1=".95 .95 .98" rgb2=".85 .85 .90" width="256" height="256"/>

    <!-- grid floor -->
    <texture name="gridtex" type="2d" builtin="checker"
             rgb1="1 1 1" rgb2="0.6 0.6 0.8"
             width="512" height="512"/>
    <!-- texture repeat count (X Y). Increasing the number makes the pattern finer -->
    <material name="gridmat" texture="gridtex" texrepeat="1 1" texuniform="true"/>
  </asset>

  <worldbody>
    <!-- ground floor with grid texture -->
    <geom name="ground" type="plane" size="20 20 0.1" material="gridmat" rgba="1 1 1 1"/>

    <!-- sun light -->
    <light name="sun" pos="0 0 5" dir="0 0 -1" directional="true"
           diffuse="1 1 1" specular="0.2 0.2 0.2" ambient="0.5 0.5 0.5"/>
  </worldbody>
</mujoco>
"""

def draw_arrow(viewer, arrow_origin, arrow_vector,
                *, scale=0.01, width=0.01, rgba=(0.9, 0.2, 0.2, 1.0), arrow_type=mujoco.mjtGeom.mjGEOM_ARROW):
    """
    arrow_origin: the origin point of arrow [3]
    arrow_vector: vector [3]
    scale:       scale
    width:       width
    """
    arrow_head = arrow_origin + arrow_vector * scale

    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom: return

    # generate an arrow geometry
    mujoco.mjv_makeConnector(
        # geom,
        scene.geoms[scene.ngeom],
        arrow_type,  # arrow type
        width,
        float(arrow_origin[0]), float(arrow_origin[1]), float(arrow_origin[2]),
        float(arrow_head[0]), float(arrow_head[1]), float(arrow_head[2])
    )
    scene.geoms[scene.ngeom].rgba[:] = rgba
    scene.ngeom += 1

def draw_frame(viewer, origin_pos, rot_mat, *, scale=0.1, width=0.01, alpha=1.0, array_type=mujoco.mjtGeom.mjGEOM_ARROW1):
    # rot_mat: 3x3 rotation matrix
    draw_arrow(viewer, origin_pos, rot_mat[:,0],
                scale=scale, width=width, rgba=(0.9, 0.2, 0.2, alpha), arrow_type=array_type) # x-axis (red)
    draw_arrow(viewer, origin_pos, rot_mat[:,1],
                scale=scale, width=width, rgba=(0.2, 0.9, 0.2, alpha), arrow_type=array_type) # y-axis (green)
    draw_arrow(viewer, origin_pos, rot_mat[:,2],
                scale=scale, width=width, rgba=(0.2, 0.2, 0.9, alpha), arrow_type=array_type) # z-axis (blue)

def draw_arc(viewer, arc_center, arc_axis, central_angle, 
            radius=0.05, nseg=12, width=0.004, rgba_pos=(0.2,0.6,1.0,1.0), rgba_neg=(1.0,0.5,0.2,1.0)):
    """
    arc_center: the origin of arc[3]
    arc_axis: the joint axis (world coordinates, normalization recommended) [3]
    central_angle: [rad] (rotation direction by sign)
    radius: arc radius
    """
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom: return

    # normalize arc_axis
    z = arc_axis / (np.linalg.norm(arc_axis) + 1e-12)
    # create u from a non-parallel vector
    tmp = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
    u = np.cross(z, tmp); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(z, u)

    # set color based on rotation direction
    rgba = rgba_pos if central_angle >= 0 else rgba_neg

    # divide the arc by nseg
    thetas = np.linspace(0.0, central_angle, nseg+1)
    pts = [arc_center + radius*(np.cos(th)*u + np.sin(th)*v) for th in thetas]

    # connect line segements
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom: break
        mujoco.mjv_makeConnector(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            float(a[0]), float(a[1]), float(a[2]),
            float(b[0]), float(b[1]), float(b[2])
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1

    # add a small arrow at the end of the arc
    if scn.ngeom < scn.maxgeom:
        end = pts[-1]
        # tangential direction (z × r)
        r_end = (end - arc_center); r_end /= (np.linalg.norm(r_end)+1e-12)
        tangent = np.cross(z, r_end)
        head = end + 0.04 * tangent  # arrow length
        mujoco.mjv_makeConnector(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, width,
            float(end[0]), float(end[1]), float(end[2]),
            float(head[0]), float(head[1]), float(head[2])
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1