"""Engine profile arg rendering (shell-split lines, placeholder substitution)."""
import pytest

from llmbench.core.engines import EngineProfile, ServerProcess


def profile(args):
    return EngineProfile(name="t", executable="exe", args=args)


class TestRenderArgs:
    def test_flag_and_value_on_one_line(self):
        # the exact shape that used to be glued into one argv element
        p = profile(["--model {model}", "--host 127.0.0.1", "--port {port}"])
        assert p.render_args("/m/o.gguf", 47275) == [
            "--model", "/m/o.gguf", "--host", "127.0.0.1", "--port", "47275",
        ]

    def test_one_token_per_line_still_works(self):
        p = profile(["--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"])
        assert p.render_args("/m/o.gguf", 1) == [
            "--model", "/m/o.gguf", "--host", "127.0.0.1", "--port", "1",
        ]

    def test_multiple_args_one_line(self):
        p = profile(["--batch-size 2048 --ubatch-size 1024"])
        assert p.render_args("", 0) == ["--batch-size", "2048", "--ubatch-size", "1024"]

    def test_quoted_value_with_spaces(self):
        p = profile(['--model "{model}"'])
        assert p.render_args("/a b/c.gguf", 0) == ["--model", "/a b/c.gguf"]

    def test_quoted_json_value(self):
        p = profile(["--chat-template-kwargs '{\"k\":true}'"])
        assert p.render_args("", 0) == ["--chat-template-kwargs", '{"k":true}']

    def test_bad_quotes_raise(self):
        p = profile(["--x 'unterminated"])
        with pytest.raises(ValueError):
            p.render_args("", 0)


class TestServerProcessEnv:
    def _server(self, profile_env, extra_env):
        prof = EngineProfile(name="t", executable="exe", args=["--x"], env=profile_env)
        return ServerProcess(prof, [], "", 0, "/tmp/x.log", extra_env=extra_env)

    def test_engine_env_applied(self):
        s = self._server({"A": "1"}, None)
        assert s.merged_env["A"] == "1"

    def test_sweep_extra_env_overrides_engine_env(self):
        s = self._server({"A": "engine", "B": "keep"}, {"A": "sweep", "C": "sweep"})
        assert s.merged_env["A"] == "sweep"
        assert s.merged_env["B"] == "keep"
        assert s.merged_env["C"] == "sweep"

    def test_inherits_process_env(self):
        s = self._server({}, {})
        # PATH always exists in the process env
        assert "PATH" in s.merged_env
