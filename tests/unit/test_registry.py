import pytest

from agent_harness import ErrorCode, FunctionTool, Tool, ToolRegistry, ValidationError


def make(name):
    return FunctionTool(lambda: None, name=name)


def test_register_lookup_and_schemas():
    registry = ToolRegistry([make("a"), make("b")])
    assert "a" in registry and len(registry) == 2
    assert registry.get("a").name == "a" and registry.get("zzz") is None
    assert registry.names() == ["a", "b"]
    assert [s.name for s in registry.schemas()] == ["a", "b"]


def test_duplicate_names_rejected():
    registry = ToolRegistry([make("a")])
    with pytest.raises(ValidationError) as info:
        registry.register(make("a"))
    assert info.value.code == ErrorCode.INVALID_CONFIG


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "dots.not.ok"])
def test_invalid_names_rejected(name):
    with pytest.raises(ValidationError):
        ToolRegistry().register(make(name))


def test_non_tool_rejected():
    with pytest.raises(ValidationError):
        ToolRegistry().register(object())  # type: ignore[arg-type]


def test_unregister():
    registry = ToolRegistry([make("a")])
    assert registry.unregister("a").name == "a"
    assert "a" not in registry
    with pytest.raises(KeyError):
        registry.unregister("a")


class Tracked(Tool):
    def __init__(self, name, log, fail_setup=False, fail_teardown=False):
        self.name, self.log, self.fail_setup, self.fail_teardown = name, log, fail_setup, fail_teardown

    async def execute(self, arguments, ctx):
        return None

    async def setup(self):
        if self.fail_setup:
            raise RuntimeError("setup failed")
        self.log.append(f"setup:{self.name}")

    async def teardown(self):
        self.log.append(f"teardown:{self.name}")
        if self.fail_teardown:
            raise RuntimeError("teardown failed")


async def test_setup_then_teardown_in_reverse_order():
    log = []
    registry = ToolRegistry([Tracked("a", log), Tracked("b", log)])
    await registry.setup()
    assert await registry.teardown() == []
    assert log == ["setup:a", "setup:b", "teardown:b", "teardown:a"]


async def test_failed_setup_tears_down_already_set_up_tools():
    log = []
    registry = ToolRegistry([Tracked("a", log), Tracked("b", log, fail_setup=True), Tracked("c", log)])
    with pytest.raises(RuntimeError):
        await registry.setup()
    assert log == ["setup:a", "teardown:a"]


async def test_teardown_errors_are_contained():
    log = []
    registry = ToolRegistry([Tracked("a", log, fail_teardown=True), Tracked("b", log)])
    await registry.setup()
    errors = await registry.teardown()
    assert len(errors) == 1
    assert log[-2:] == ["teardown:b", "teardown:a"]
