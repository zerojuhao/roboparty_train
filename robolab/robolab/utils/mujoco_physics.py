"""Apply exported Newton collision assets to CPU MuJoCo, independently of policies.

Profiles contain compiled geometry/contact/solver data, never checkpoint or motor
snapshots. Motor limits are resolved from the URDF and the supplied actuator config.
"""

from functools import cached_property
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

CONTACT_FIELDS = ("friction", "solref", "solimp", "margin", "gap", "condim", "contype",
                  "conaffinity", "priority", "solmix")
OPTION_FIELDS = ("timestep", "integrator", "solver", "cone", "impratio", "iterations",
                 "ls_iterations", "tolerance", "ccd_iterations", "ccd_tolerance",
                 "disableflags", "enableflags", "gravity")


def xml_value(value):
    return " ".join(str(x) for x in np.asarray(value).reshape(-1).tolist())


class NewtonPhysicsAsset:
    """A robot's nominal compiled physics, with explicit asset location and roots."""

    prefix = "newton_native_"

    def __init__(self, path, *, data_root):
        self.path = Path(path)
        self.data_root = Path(data_root)

    @cached_property
    def profile(self):
        profile = json.loads(self.path.read_text())
        if profile["schema"] != 2:
            raise ValueError("Unsupported Newton physics asset schema")
        for relative, digest in profile["sources"].items():
            path = self.data_root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Physics asset is stale for {path}; re-export the nominal model")
        disabled = int(profile["options"]["disableflags"])
        if disabled & (int(mujoco.mjtDisableBit.mjDSBL_LIMIT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
                       | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
            raise ValueError("Physics asset must retain contacts and joint constraints")
        return profile

    @property
    def urdf(self):
        return self.data_root / self.profile["urdf"]

    def align_robot_xml(self, robot):
        profile = self.profile
        asset = robot.find("asset")
        if asset is None:
            asset = ET.SubElement(robot, "asset")
        bodies = {body.get("name"): body for body in robot.findall(".//body")}
        missing = {g["body"] for g in profile["geometries"]} - {"world"} - bodies.keys()
        if missing:
            raise ValueError(f"MJCF is missing physics asset bodies: {sorted(missing)}")
        # Retain visual meshes, kinematics and inertias for rendering and FK parity.
        for body in bodies.values():
            for geom in list(body.findall("geom")):
                if geom.get("class") == "collision":
                    body.remove(geom)
        for index, geom in enumerate(profile["geometries"]):
            if geom["body"] == "world":
                continue
            name = self.prefix + str(index)
            attrs = {key: xml_value(geom[key]) for key in CONTACT_FIELDS}
            attrs.update(name=name, type=geom["type"], group="3",
                         pos=xml_value(geom["pos"]), quat=xml_value(geom["quat"]))
            if geom["type"] == "mesh":
                ET.SubElement(asset, "mesh", name=name, vertex=xml_value(geom["vertices"]),
                              face=xml_value(geom["faces"]))
                attrs["mesh"] = name
            else:
                attrs["size"] = xml_value(geom["size"])
            ET.SubElement(bodies[geom["body"]], "geom", **attrs)
        contact = robot.find("contact")
        if contact is not None:
            robot.remove(contact)
        contact = ET.SubElement(robot, "contact")
        for a, b in profile["excludes"]:
            ET.SubElement(contact, "exclude", body1=a, body2=b)
        joints = {joint.get("name"): joint for joint in robot.findall(".//joint")}
        for name, params in profile["joints"].items():
            if name not in joints:
                raise ValueError(f"MJCF is missing physics asset joint: {name}")
            joint = joints[name]
            if not np.allclose(np.fromstring(joint.get("range", ""), sep=" "), params["range"], atol=1e-6):
                raise ValueError(f"Physical ranges differ for {name}")
            joint.set("limited", "true" if params["limited"] else "false")
            joint.set("solreflimit", xml_value(params["solref"]))
            joint.set("solimplimit", xml_value(params["solimp"]))
            joint.set("margin", xml_value(params["margin"]))

    def is_native_model(self, model):
        return any(model.geom(g).name.startswith(self.prefix) for g in range(model.ngeom))

    def align_compiled_model(self, model):
        if not self.is_native_model(model):
            return
        for key, value in self.profile["options"].items():
            setattr(model.opt, key, value)
        floor = next(g for g in self.profile["geometries"] if g["body"] == "world")
        # MjSpec attaches terrain after robot XML preparation.
        for g in range(model.ngeom):
            if model.body_weldid[model.geom_bodyid[g]] == 0 and (model.geom_contype[g] or model.geom_conaffinity[g]):
                for key in CONTACT_FIELDS:
                    getattr(model, "geom_" + key)[g] = floor[key]


def export_physics_asset(model, urdf, output, *, data_root, timestep):
    """Export one nominal robot and a plane from a compiled model, without a policy.

    Use a nonrandomized single-environment Newton model. The caller supplies the
    actual substep dt because Newton's CPU shadow model can retain a default dt.
    """
    urdf, data_root = Path(urdf).resolve(), Path(data_root).resolve()
    root = ET.parse(urdf).getroot()
    names = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    body_names = [body.get("name") for body in root.findall("link")]

    def canonical(name, candidates):
        if name in candidates:
            return name
        matches = [candidate for candidate in candidates if name.endswith("/" + candidate)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or unknown model name: {name}: {matches}")
        return matches[0]

    bodies = {i: canonical(model.body(i).name, body_names) for i in range(1, model.nbody)}
    if len(set(bodies.values())) != len(bodies):
        raise ValueError("Export requires a single robot, not multiple environment copies")
    geometries = []
    types = {int(getattr(mujoco.mjtGeom, "mjGEOM_" + name.upper())): name
             for name in ("mesh", "cylinder", "capsule", "box", "sphere", "plane", "ellipsoid")}
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        item = {key: np.asarray(getattr(model, "geom_" + key)[i]).tolist() for key in CONTACT_FIELDS}
        item.update(body=bodies.get(int(model.geom_bodyid[i]), "world"), type=types[int(model.geom_type[i])],
                    pos=model.geom_pos[i].tolist(), quat=model.geom_quat[i].tolist(), size=model.geom_size[i].tolist())
        if item["type"] == "mesh":
            mesh = model.geom_dataid[i]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[i])
            item["vertices"] = (model.mesh_vert[start:start + count].astype(float) @ rotation.reshape(3, 3).T
                                + model.geom_pos[i]).tolist()
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            item["faces"] = model.mesh_face[start:start + count].tolist()
            item["pos"], item["quat"] = [0., 0., 0.], [1., 0., 0., 0.]
        geometries.append(item)
    world = [g for g in geometries if g["body"] == "world"]
    if len(world) != 1 or world[0]["type"] != "plane":
        raise ValueError("Export requires one nominal ground plane, not a training terrain")
    joints = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = canonical(model.joint(j).name, names)
        joints[name] = {key: np.asarray(getattr(model, "jnt_" + key)[j]).tolist()
                        for key in ("limited", "range", "solref", "solimp", "margin")}
    if set(joints) != set(names):
        raise ValueError("Model joints do not match the URDF")
    paths = {urdf}
    paths.update((urdf.parent / mesh.get("filename")).resolve() for mesh in root.findall(".//mesh"))
    sources = {str(path.relative_to(data_root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(paths)}
    options = {key: np.asarray(getattr(model.opt, key)).tolist() for key in OPTION_FIELDS}
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("Physics timestep must be positive and finite")
    options["timestep"] = timestep
    profile = dict(schema=2, mujoco_version=mujoco.__version__, urdf=str(urdf.relative_to(data_root)),
                   sources=sources, joint_names=names, joints=joints, geometries=geometries,
                   excludes=[[bodies[int(sig) >> 16], bodies[int(sig) & 65535]] for sig in model.exclude_signature],
                   options=options)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    return profile
