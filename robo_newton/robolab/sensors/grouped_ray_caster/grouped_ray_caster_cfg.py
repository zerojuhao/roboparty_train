from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/Geometry/.*",
    suffix: str | None = None,
    is_shared=True,  # whether the target prim is assumed to be the same mesh across all environments.
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Build the raycast target given the list of links. It will combine and return a list of
    MultiMeshRayCasterCfg.RaycastTargetCfg.
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=f"{prefix}{link}{suffix if suffix is not None else '/' + link}",
            is_shared=is_shared,
            **kwargs,
        )
        for link in links
    ]
