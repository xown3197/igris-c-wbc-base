"""Narrow compatibility fixes for the hash-locked manufacturer public source."""

from __future__ import annotations


def install_action_term_cfg_alias() -> bool:
    """Restore the import path used by manufacturer ``mdp/actions.py``.

    IsaacLab f4aa17 defines ``ActionTermCfg`` in ``isaaclab.managers`` but does
    not re-export it from ``isaaclab.envs.mdp.actions``.  The manufacturer
    package imports the missing re-export.  Installing this alias changes no
    environment, physics, or observation behavior.
    """

    import isaaclab.envs.mdp.actions as action_module
    from isaaclab.managers import ActionTermCfg

    current = getattr(action_module, "ActionTermCfg", None)
    if current is None:
        action_module.ActionTermCfg = ActionTermCfg
        return True
    if current is not ActionTermCfg:
        raise RuntimeError("unexpected ActionTermCfg already exists at legacy path")
    return False


__all__ = ["install_action_term_cfg_alias"]
