"""Recommendation / interpretation logic and env placeholder substitution."""
from llmbench.core.engines import EngineProfile, ServerProcess, builtin_presets
from llmbench.core.report import recommendations


def _numa_rows():
    # Mirrors a real llama.cpp --numa sweep (pg1024,512).
    def row(label, pp, tg, e2e, e2e_s, p99, p99_s):
        return dict(
            workload="pg1024,512", label=label, n=3,
            pp_tps_mean=pp, pp_tps_std=1.0, tg_tps_mean=tg, tg_tps_std=1.0,
            ttft_ms_mean=130.0, ttft_ms_std=0.1, e2e_ms_mean=e2e, e2e_ms_std=e2e_s,
            itl_p50_ms_mean=0.1, itl_p50_ms_std=0.01, itl_p90_ms_mean=60.0, itl_p90_ms_std=1.0,
            itl_p99_ms_mean=p99, itl_p99_ms_std=p99_s,
        )
    return [
        row("numa=distribute", 8656.76, 48.44, 10875.64, 1726.28, 67.09, 0.46),
        row("numa=isolate", 8629.52, 43.00, 12025.83, 435.40, 67.42, 0.15),
        row("numa=numactl", 8339.05, 95.87, 5599.57, 1089.81, 229.09, 121.19),
    ]


class TestRecommend:
    def test_picks_fastest_e2e(self):
        recs = recommendations(_numa_rows())
        assert len(recs) == 1
        assert recs[0]["main"] == "numa=numactl"
        assert recs[0]["main_by"] == "total time (E2E)"

    def test_flags_tail_latency_caveat(self):
        recs = recommendations(_numa_rows())
        text = " ".join(recs[0]["caveats"])
        assert "tail latency" in text
        # the consistent variant is called out as the low-latency pick
        assert "numa=distribute" in text or "numa=isolate" in text

    def test_flags_unstable_variants(self):
        recs = recommendations(_numa_rows())
        assert "numa=numactl" in recs[0]["unstable"]
        # isolate is stable (small e2e/p99 spread) so should not be flagged
        assert "numa=isolate" not in recs[0]["unstable"]

    def test_clean_single_winner_no_caveats(self):
        rows = [
            dict(workload="tg64", label="a", n=3, tg_tps_mean=100.0, tg_tps_std=1.0,
                 e2e_ms_mean=1000.0, e2e_ms_std=5.0, itl_p99_ms_mean=10.0, itl_p99_ms_std=0.2),
            dict(workload="tg64", label="b", n=3, tg_tps_mean=50.0, tg_tps_std=1.0,
                 e2e_ms_mean=2000.0, e2e_ms_std=5.0, itl_p99_ms_mean=20.0, itl_p99_ms_std=0.2),
        ]
        recs = recommendations(rows)
        assert recs[0]["main"] == "a"
        assert recs[0]["caveats"] == []
        assert recs[0]["unstable"] == []


class TestEnvSubstitution:
    def _server(self, env, extra):
        prof = EngineProfile(name="t", executable="exe", args=["--x"], env=env)
        return ServerProcess(prof, [], "/m/o.gguf", 41000, "/tmp/x.log", extra_env=extra)

    def test_substitutes_port_and_model(self):
        s = self._server({"OLLAMA_HOST": "127.0.0.1:{port}", "M": "{model}"}, {})
        assert s.merged_env["OLLAMA_HOST"] == "127.0.0.1:41000"
        assert s.merged_env["M"] == "/m/o.gguf"

    def test_extra_env_overrides_and_substitutes(self):
        s = self._server({"A": "1"}, {"A": "x-{port}"})
        assert s.merged_env["A"] == "x-41000"


class TestPresets:
    def test_all_presets_have_docs(self):
        for p in builtin_presets():
            assert p.get("docs", "").startswith("http"), f"{p['name']} missing docs link"

    def test_known_engines_present(self):
        names = {p["name"] for p in builtin_presets()}
        for want in ("llama.cpp (llama-server)", "vLLM", "KoboldCpp", "Ollama", "SGLang"):
            assert want in names


class TestDefaultEngineOrdering:
    def test_default_engine_is_first_in_api(self, tmp_path):
        from fastapi.testclient import TestClient
        from llmbench.app import create_app
        from llmbench.core.engines import DEFAULT_ENGINE
        app, _, _ = create_app(tmp_path / "data")
        with TestClient(app) as client:  # startup seeds the presets
            names = [e["name"] for e in client.get("/api/engines").json()]
        assert names[0] == DEFAULT_ENGINE

