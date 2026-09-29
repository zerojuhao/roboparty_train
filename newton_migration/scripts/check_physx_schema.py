"""Probe PhysX USD schema registration needed by the URDF importer."""

from pathlib import Path

from pxr import Plug, Usd

import ovphysx

schema_root = Path(ovphysx.__file__).parent / "plugins" / "usd" / "PhysxSchema" / "resources"
print(f"schema={schema_root}")
print(f"registered={Plug.Registry().RegisterPlugins(str(schema_root / 'plugInfo.json'))}")
stage = Usd.Stage.CreateInMemory()
prim = stage.DefinePrim("/Joint")
print(f"applied={prim.ApplyAPI('PhysxJointAPI')}")
