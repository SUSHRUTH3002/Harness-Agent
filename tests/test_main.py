"""main.py configuration: .env loading rules and config validation. No network."""

import os

import pytest

pytest.importorskip("litellm")
import main  # noqa: E402

# Every key in .env.example: litellm's import-time load_dotenv() may have exported the developer's real .env.
KEYS = [
    "HARNESS_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    "HARNESS_LOCAL_MODEL", "HARNESS_API_BASE", "HARNESS_API_KEY", "HARNESS_TEMPERATURE", "HARNESS_MAX_TOKENS",
    "HARNESS_MAX_STEPS", "HARNESS_TIMEOUT", "HARNESS_LOG_LEVEL", "HARNESS_LOG_FILE",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(main, "ROOT", tmp_path)  # load_config reads ROOT/.env
    return tmp_path


def write_env(path, text):
    (path / ".env").write_text(text)


def test_values_are_loaded_and_blanks_skipped(clean_env):
    write_env(clean_env, "HARNESS_MODEL=openai/m\nOPENAI_API_KEY=sk-x\nANTHROPIC_API_KEY=\n")
    main.load_env_file(clean_env / ".env")
    assert os.environ["HARNESS_MODEL"] == "openai/m" and os.environ["OPENAI_API_KEY"] == "sk-x"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_blank_entries_already_exported_as_empty_are_removed(clean_env, monkeypatch):
    # What litellm's import-time load_dotenv() does with an unfilled template line.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    write_env(clean_env, "ANTHROPIC_API_KEY=\n")
    main.load_env_file(clean_env / ".env")
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_shell_values_win(clean_env, monkeypatch):
    monkeypatch.setenv("HARNESS_MODEL", "anthropic/from-shell")
    write_env(clean_env, "HARNESS_MODEL=openai/from-file\n")
    main.load_env_file(clean_env / ".env")
    assert os.environ["HARNESS_MODEL"] == "anthropic/from-shell"


def test_missing_model_is_a_config_error(clean_env):
    write_env(clean_env, "")
    with pytest.raises(main.ConfigError, match="HARNESS_MODEL"):
        main.load_config()


def test_missing_provider_key_is_caught_before_any_request(clean_env):
    write_env(clean_env, "HARNESS_MODEL=anthropic/some-model\nANTHROPIC_API_KEY=\n")
    with pytest.raises(main.ConfigError, match="ANTHROPIC_API_KEY"):
        main.load_config()


def test_custom_endpoint_skips_key_check_and_parses_numbers(clean_env):
    write_env(clean_env, "HARNESS_MODEL=ollama/m\nHARNESS_API_BASE=http://localhost:11434\nHARNESS_MAX_STEPS=7\n")
    config = main.load_config()
    assert (config["model"], config["api_base"], config["max_steps"]) == ("ollama/m", "http://localhost:11434", 7)
    assert config["local"] is False


def test_invalid_number_is_a_config_error(clean_env):
    write_env(clean_env, "HARNESS_MODEL=ollama/m\nHARNESS_API_BASE=http://x\nHARNESS_MAX_STEPS=abc\n")
    with pytest.raises(main.ConfigError, match="HARNESS_MAX_STEPS"):
        main.load_config()


def test_temperature_is_unset_by_default(clean_env):
    write_env(clean_env, "HARNESS_MODEL=ollama/m\nHARNESS_API_BASE=http://x\n")
    assert "temperature" not in main.load_config()["settings"]


def test_temperature_is_read_when_set(clean_env):
    write_env(clean_env, "HARNESS_MODEL=ollama/m\nHARNESS_API_BASE=http://x\nHARNESS_TEMPERATURE=0.2\n")
    assert main.load_config()["settings"]["temperature"] == 0.2


def test_local_model_requires_api_base(clean_env):
    write_env(clean_env, "HARNESS_MODEL=llama-3.1-8b\nHARNESS_LOCAL_MODEL=true\n")
    with pytest.raises(main.ConfigError, match="HARNESS_API_BASE"):
        main.load_config()


def test_local_model_skips_the_litellm_key_check(clean_env):
    write_env(
        clean_env,
        "HARNESS_MODEL=llama-3.1-8b\nHARNESS_LOCAL_MODEL=true\nHARNESS_API_BASE=http://localhost:11434/v1\n",
    )
    config = main.load_config()
    assert (config["local"], config["model"], config["api_base"]) == (True, "llama-3.1-8b", "http://localhost:11434/v1")


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_bool_parsing_accepts_common_truthy_spellings(clean_env, value):
    write_env(clean_env, f"HARNESS_MODEL=m\nHARNESS_LOCAL_MODEL={value}\nHARNESS_API_BASE=http://x\n")
    assert main.load_config()["local"] is True


def test_bool_parsing_rejects_other_values(clean_env):
    write_env(clean_env, "HARNESS_MODEL=m\nHARNESS_LOCAL_MODEL=maybe\n")
    with pytest.raises(main.ConfigError, match="HARNESS_LOCAL_MODEL"):
        main.load_config()


def test_build_selects_local_provider(clean_env):
    write_env(clean_env, "HARNESS_MODEL=m\nHARNESS_LOCAL_MODEL=true\nHARNESS_API_BASE=http://localhost:11434/v1\n")
    runtime, _ = main.build(main.load_config())
    assert type(runtime.llm).__name__ == "LocalProvider"


def test_build_selects_litellm_provider_by_default(clean_env):
    write_env(clean_env, "HARNESS_MODEL=ollama/m\nHARNESS_API_BASE=http://localhost:11434\n")
    runtime, _ = main.build(main.load_config())
    assert type(runtime.llm).__name__ == "LiteLLMProvider"


def test_noisy_third_party_loggers_are_capped_at_warning(clean_env):
    import logging

    # setup_logging() mutates global logging state (root handlers, named-logger levels),
    # which would otherwise leak into every test that runs after this one in the same session.
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    watched = (*main._NOISY_LIBRARIES, "agent_harness", "main")
    saved_levels = {name: logging.getLogger(name).level for name in watched}
    try:
        main.setup_logging("DEBUG", None)
        for name in main._NOISY_LIBRARIES:
            assert logging.getLogger(name).level == logging.WARNING
        assert logging.getLogger("agent_harness").level == logging.DEBUG
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
