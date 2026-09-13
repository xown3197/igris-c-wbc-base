import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_verified_upstream.py"
SPEC = importlib.util.spec_from_file_location("run_verified_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_injection_occurs_exactly_after_manufacturer_task_import():
    source = "before\nimport robros_lab_public.tasks  # noqa: F401\nafter\n"
    injected = MODULE.inject_wbc_tasks(source)
    assert injected == (
        "before\n"
        "from wbc_base.compat import install_action_term_cfg_alias\n"
        "install_action_term_cfg_alias()\n"
        "import robros_lab_public.tasks  # noqa: F401\n"
        "import wbc_base.tasks  # noqa: F401\n"
        "after\n"
    )


@pytest.mark.parametrize("source", ["", "import wbc_base.tasks  # noqa: F401"])
def test_injection_refuses_unexpected_source(source):
    with pytest.raises(RuntimeError):
        MODULE.inject_wbc_tasks(source)
