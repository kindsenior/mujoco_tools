import mujoco
import numpy as np

def draw_arrow(viewer, arrow_origin, arrow_vector, scale=0.01, width=0.01, rgba=(0.9, 0.2, 0.2, 1.0)):
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
        mujoco.mjtGeom.mjGEOM_ARROW,  # 矢印
        width,
        float(arrow_origin[0]), float(arrow_origin[1]), float(arrow_origin[2]),
        float(arrow_head[0]), float(arrow_head[1]), float(arrow_head[2])
    )
    scene.geoms[scene.ngeom].rgba[:] = rgba
    scene.ngeom += 1

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