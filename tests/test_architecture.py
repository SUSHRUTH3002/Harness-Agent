"""Architecture guards: the core depends only on the standard library and pydantic."""

import ast
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "agent_harness"
ALLOWED_THIRD_PARTY = {"pydantic"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


OPTIONAL_SUBPACKAGES = {"testing", "providers"}


def core_files():
    return [p for p in CORE.rglob("*.py") if not OPTIONAL_SUBPACKAGES & set(p.relative_to(CORE).parts)]


def test_core_has_no_unexpected_dependencies():
    for path in [p for p in CORE.rglob("*.py") if "providers" not in p.relative_to(CORE).parts]:
        for module in imports_of(path):
            top = module.split(".")[0]
            assert top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY or top == "agent_harness", (
                f"{path.relative_to(CORE)} imports {module}"
            )


def test_core_does_not_import_testing_utilities_or_providers():
    for path in core_files():
        for module in imports_of(path):
            assert not module.startswith(("agent_harness.testing", "agent_harness.providers")), (
                f"{path.relative_to(CORE)} imports {module}"
            )


def test_importing_the_package_does_not_load_provider_sdks():
    import subprocess

    code = "import sys, agent_harness; assert 'litellm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_loop_does_not_import_the_runtime():
    assert not any(m.startswith("agent_harness.runtime") for m in imports_of(CORE / "loop.py"))
