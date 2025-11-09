from typing import Any, Dict, List, Optional, Tuple
from pxr import Usd, UsdGeom, Sdf
from newton._src.geometry.flags import ShapeFlags



def _get_single_rel_target(prim, rel_name: str) -> Optional[Sdf.Path]:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return targets[0]

# def _capsule_height_for_body(stage, body_path: Sdf.Path) -> float:
#     # candidate 1: explicit "Geom" child
#     p_geom = stage.GetPrimAtPath(body_path.AppendPath("Geom"))
#     if p_geom and p_geom.IsValid() and p_geom.GetTypeName() == "Capsule":
#         h = UsdGeom.Capsule(p_geom).GetHeightAttr().Get()
#         return float(h) if h is not None else 0.0
#     # candidate 2: any direct child of type Capsule
#     body_prim = stage.GetPrimAtPath(body_path)
#     if body_prim and body_prim.IsValid():
#         for child in body_prim.GetChildren():
#             if child.GetTypeName() == "Capsule":
#                 h = UsdGeom.Capsule(child).GetHeightAttr().Get()
#                 return float(h) if h is not None else 0.0
#     # fallback
#     return 0.0
def _capsule_height_for_body(stage, body_path: Sdf.Path) -> float:
    # Helper: get height from any known shape
    def _get_height(prim):
        type_name = prim.GetTypeName()
        if type_name == "Capsule":
            return UsdGeom.Capsule(prim).GetHeightAttr().Get()
        elif type_name == "Cylinder":
            return UsdGeom.Cylinder(prim).GetHeightAttr().Get()
        else:
            return None

    # candidate 1: explicit "Geom" child
    p_geom = stage.GetPrimAtPath(body_path.AppendPath("Geom"))
    if p_geom and p_geom.IsValid():
        h = _get_height(p_geom)
        if h is not None:
            return float(h)

    # candidate 2: any direct child of type Capsule/Cylinder
    body_prim = stage.GetPrimAtPath(body_path)
    if body_prim and body_prim.IsValid():
        for child in body_prim.GetChildren():
            h = _get_height(child)
            if h is not None:
                return float(h)

    # fallback
    return 0.0

def _read_rest_quat_xyzw(stage, body_path: Sdf.Path) -> Tuple[float, float, float, float]:
    """Read custom attr newton:restQuat as [x, y, z, w]; default to (0,0,0,1)."""
    prim = stage.GetPrimAtPath(body_path)
    attr = prim.GetAttribute("newton:restQuat")
    val = attr.Get()
    return (float(val[0]), float(val[1]), float(val[2]), float(val[3]))


def parse_rod_constraint(
    source: str,    # path to USD file/layer
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      - name
      - prim_path
      - body0_path
      - body1_path
      - body0_length   # NEW: Capsule height of body0's Geom
      - body1_length   # NEW: Capsule height of body1's Geom
    """
    stage = Usd.Stage.Open(source)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {source}")

    out: List[Dict[str, Any]] = []

    for prim in stage.Traverse():
        if not prim.IsValid() or prim.GetTypeName() != "RodConstraint":
            continue

        body0 = _get_single_rel_target(prim, "physics:body0")
        body1 = _get_single_rel_target(prim, "physics:body1")
        if not (body0 and body1):
            continue

        # --- find Capsule height for each body ---
        # Try "<body>/Geom" first; if missing, search direct children for type "Capsule"


        body0_len = _capsule_height_for_body(stage, body0)
        body1_len = _capsule_height_for_body(stage, body1)

        b0_qx, b0_qy, b0_qz, b0_qw = _read_rest_quat_xyzw(stage, body0)
        b1_qx, b1_qy, b1_qz, b1_qw = _read_rest_quat_xyzw(stage, body1)


        out.append({
            "name": prim.GetName(),
            "prim_path": prim.GetPath(),
            "body0_path": body0,
            "body1_path": body1,
            "body0_length": body0_len,
            "body1_length": body1_len,
            "body0_rest_quat": (b0_qx, b0_qy, b0_qz, b0_qw),
            "body1_rest_quat": (b1_qx, b1_qy, b1_qz, b1_qw),
        })

    return out


def update_joint(model, rod_constraints):
    import numpy as np
    import warp as wp

    if not rod_constraints:
        return

    dev = model.device
    ROD_CONSTRAINT = 7
    DOF_PER = 6

    # ---- read existing as numpy (or init empty) ----
    joint_type        = model.joint_type.numpy()        
    joint_enabled     = model.joint_enabled.numpy()    
    joint_parent      = model.joint_parent.numpy()  
    joint_child       = model.joint_child.numpy() 
    joint_X_p         = model.joint_X_p.numpy()
    joint_X_c         = model.joint_X_c.numpy()
    joint_limit_lower = model.joint_limit_lower.numpy()
    joint_limit_upper = model.joint_limit_upper.numpy()
    joint_qd_start    = model.joint_qd_start.numpy()
    joint_dof_dim     = model.joint_dof_dim.numpy()
    joint_dof_mode    = model.joint_dof_mode.numpy()
    joint_axis        = model.joint_axis.numpy()
    joint_target_ke   = model.joint_target_ke.numpy() 
    joint_target_kd   = model.joint_target_kd.numpy()

    joint_key = list(getattr(model, "joint_key", []) or [])

    # If qd_start empty, seed from current dof count (length of per-DOF arrays)
    if joint_qd_start.size == 0:
        joint_qd_start = np.array([joint_axis.shape[0]], dtype=np.int32)

    # ---- prepare new rows (collect in lists, concat once) ----
    new_joint_type, new_joint_enabled = [], []
    new_joint_parent, new_joint_child = [], []
    new_joint_X_p, new_joint_X_c = [], []
    new_joint_dof_dim = []

    new_limit_lo, new_limit_hi = [], []
    new_dof_mode = []
    new_axis = []
    new_ke, new_kd = [], []
    new_keys = []

    identity7 = np.array([0.,0.,0.,0.,0.,0.,1.], dtype=np.float32)
    axes6 = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    lim_lo6 = np.full((DOF_PER,), -1e6, dtype=np.float32)
    lim_hi6 = np.full((DOF_PER,),  1e6, dtype=np.float32)
    mode6   = np.zeros((DOF_PER,), dtype=np.int32)
    ke6     = np.zeros((DOF_PER,), dtype=np.float32)
    kd6     = np.zeros((DOF_PER,), dtype=np.float32)

    # convenience for body lookup: allow exact string or basename match
    body_keys = list(getattr(model, "body_key", []) or [])

    def _find_body_idx(path_like):
        s = str(path_like)
        if s in body_keys:
            return body_keys.index(s)
        base = s.rsplit("/", 1)[-1]
        for i, k in enumerate(body_keys):
            if k.rsplit("/", 1)[-1] == base:
                return i
        raise ValueError(f"Body '{s}' not found in model.body_key")

    dof_cursor = joint_axis.shape[0]  # current total dof
    qd_starts_to_append = []

    rod_lengths = [[0.0, 0.0] for _ in range(joint_type.shape[0])]

    # --- NEW: accumulate per-body rest quats (xyzw) from constraints ---
    # Initialize with identity quaternion
    body_rest_quat = [(0.0, 0.0, 0.0, 1.0) for _ in range(len(body_keys))]

    for rc in rod_constraints:
        new_joint_type.append(ROD_CONSTRAINT)
        new_joint_enabled.append(1)
        new_keys.append(rc["name"])

        p = _find_body_idx(rc["body0_path"])
        c = _find_body_idx(rc["body1_path"])
        new_joint_parent.append(p)
        new_joint_child.append(c)

        # new_joint_X_p.append(identity7)
        # new_joint_X_c.append(identity7)
        new_joint_dof_dim.append([3,3])

        # per-DOF appends
        new_limit_lo.append(lim_lo6)
        new_limit_hi.append(lim_hi6)
        new_dof_mode.append(mode6)
        new_axis.append(axes6)
        new_ke.append(ke6)
        new_kd.append(kd6)

        qd_starts_to_append.append(dof_cursor)
        dof_cursor += DOF_PER
        
        b0_len = rc.get("body0_length")
        b1_len = rc.get("body1_length")
        rod_lengths.append([b0_len, b1_len])

        b0_len = rc.get("body0_length", 0.0)
        b1_len = rc.get("body1_length", 0.0)
        rod_lengths.append([b0_len, b1_len])

        hx0 = 0.5 * b0_len
        hx1 = 0.5 * b1_len
        parent_xform = np.array([0.0, 0.0, -hx0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        child_xform  = np.array([0.0, 0.0,  hx1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        new_joint_X_p.append(parent_xform)
        new_joint_X_c.append(child_xform)

        # NEW: stash rest quats per body (xyzw)
        if "body0_rest_quat" in rc:
            body_rest_quat[p] = tuple(rc["body0_rest_quat"])
        if "body1_rest_quat" in rc:
            body_rest_quat[c] = tuple(rc["body1_rest_quat"])

    # ---- concatenate with existing ----
    if new_joint_type:
        joint_type        = np.concatenate([joint_type,        np.asarray(new_joint_type, dtype=np.int32)], axis=0)
        joint_enabled     = np.concatenate([joint_enabled,     np.asarray(new_joint_enabled, dtype=np.int32)], axis=0)
        joint_parent      = np.concatenate([joint_parent,      np.asarray(new_joint_parent, dtype=np.int32)],  axis=0)
        joint_child       = np.concatenate([joint_child,       np.asarray(new_joint_child, dtype=np.int32)],   axis=0)
        joint_X_p         = np.concatenate([joint_X_p,         np.asarray(new_joint_X_p, dtype=np.float32)],  axis=0)
        joint_X_c         = np.concatenate([joint_X_c,         np.asarray(new_joint_X_c, dtype=np.float32)],  axis=0)
        joint_dof_dim     = np.concatenate([joint_dof_dim,     np.asarray(new_joint_dof_dim, dtype=np.int32)],axis=0)

        joint_limit_lower = np.concatenate([joint_limit_lower, np.asarray(new_limit_lo, dtype=np.float32).reshape(-1)], axis=0)
        joint_limit_upper = np.concatenate([joint_limit_upper, np.asarray(new_limit_hi, dtype=np.float32).reshape(-1)], axis=0)
        joint_dof_mode    = np.concatenate([joint_dof_mode,    np.asarray(new_dof_mode, dtype=np.int32).reshape(-1)],  axis=0)
        joint_axis        = np.concatenate([joint_axis,        np.asarray(new_axis, dtype=np.float32).reshape(-1,3)],  axis=0)
        joint_target_ke   = np.concatenate([joint_target_ke,   np.asarray(new_ke, dtype=np.float32).reshape(-1)],      axis=0)
        joint_target_kd   = np.concatenate([joint_target_kd,   np.asarray(new_kd, dtype=np.float32).reshape(-1)],      axis=0)

        # joint_qd_start is sentinel-sized: len = joint_count + 1
        joint_qd_start = np.concatenate([joint_qd_start, np.asarray(qd_starts_to_append, dtype=np.int32)], axis=0)
        joint_qd_start = np.concatenate([joint_qd_start, np.array([dof_cursor], dtype=np.int32)], axis=0)

        joint_key.extend(new_keys)

        rod_lengths = np.asarray(rod_lengths, dtype=np.float32)  # shape [n_existing + n_new, 2]

    # ---- NEW: build body_rest_transform [tx,ty,tz,qx,qy,qz,qw] per body ----
    body_rest_transform = np.zeros((len(body_keys), 7), dtype=np.float32)
    # zero translation already; fill quats:
    for i, q in enumerate(body_rest_quat):
        qx, qy, qz, qw = q
        body_rest_transform[i, 3:] = (qx, qy, qz, qw)

    # ---- write back as wp.array on the right device/dtypes ----
    model.joint_type        = wp.array(joint_type,        dtype=int,          device=dev)
    model.joint_enabled     = wp.array(joint_enabled,     dtype=int,          device=dev)
    model.joint_parent      = wp.array(joint_parent,      dtype=int,          device=dev)
    model.joint_child       = wp.array(joint_child,       dtype=int,          device=dev)
    model.joint_X_p         = wp.array(joint_X_p,         dtype=wp.transform, device=dev)
    model.joint_X_c         = wp.array(joint_X_c,         dtype=wp.transform, device=dev)
    model.joint_limit_lower = wp.array(joint_limit_lower, dtype=float,        device=dev)
    model.joint_limit_upper = wp.array(joint_limit_upper, dtype=float,        device=dev)
    model.joint_qd_start    = wp.array(joint_qd_start,    dtype=int,          device=dev)
    model.joint_dof_dim     = wp.array(joint_dof_dim,     dtype=int,          device=dev)
    model.joint_dof_mode    = wp.array(joint_dof_mode,    dtype=int,          device=dev)
    model.joint_axis        = wp.array(joint_axis,        dtype=wp.vec3,      device=dev)
    model.joint_target_ke   = wp.array(joint_target_ke,   dtype=float,        device=dev)
    model.joint_target_kd   = wp.array(joint_target_kd,   dtype=float,        device=dev)

    model.joint_key = joint_key
    model.joint_count = model.joint_type.shape[0]
    model.joint_dof_count = model.joint_axis.shape[0]
    model.rod_length = wp.array(rod_lengths, dtype=wp.vec2, device=dev)

    model.body_rest_transform = wp.array(body_rest_transform, dtype=wp.transform, device=dev)
    
    # ---- collision filtering (CPU-side check, merge with existing)
    shape_flags_host = model.shape_flags.numpy()

    existing_pairs = []
    if hasattr(model, "shape_collision_filter_pairs") and model.shape_collision_filter_pairs is not None:
        try:
            existing_pairs = model.shape_collision_filter_pairs.numpy().tolist()
        except Exception:
            existing_pairs = list(model.shape_collision_filter_pairs)

    new_pairs = []

    if p > -1:
        for child_shape in model.body_shapes[c]:
            if not shape_flags_host[child_shape] & ShapeFlags.COLLIDE_SHAPES:
                continue
            for parent_shape in model.body_shapes[p]:
                if not shape_flags_host[parent_shape] & ShapeFlags.COLLIDE_SHAPES:
                    continue
                a, b = parent_shape, child_shape
                if a > b:
                    a, b = b, a
                new_pairs.append((a, b))

    all_pairs = existing_pairs + new_pairs
    all_pairs = list({tuple(sorted(pair)) for pair in all_pairs})
    model.shape_collision_filter_pairs = wp.array(all_pairs, dtype=wp.vec2i, device=dev)

                    
