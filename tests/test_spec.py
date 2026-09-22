import math

import pytest

from llmbench.core.metrics import mean, percentile, stdev, summarize
from llmbench.core.spec import (
    Dimension,
    SweepSpec,
    Workload,
    expand_values,
    parse_int_range,
    plan_variants,
)


class TestRanges:
    def test_single(self):
        assert parse_int_range("8") == [8]

    def test_negative(self):
        assert parse_int_range("-1") == [-1]

    def test_span(self):
        assert parse_int_range("2-6") == [2, 3, 4, 5, 6]

    def test_step(self):
        assert parse_int_range("1-8+2") == [1, 3, 5, 7]

    def test_mult(self):
        assert parse_int_range("1-32*2") == [1, 2, 4, 8, 16, 32]

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_int_range("abc")


class TestExpandValues:
    def test_comma_list_of_ranges(self):
        assert expand_values("1-3,10") == ["1", "2", "3", "10"]

    def test_literals(self):
        assert expand_values("on,off,auto") == ["on", "off", "auto"]

    def test_semicolon_separates_values_with_commas(self):
        assert expand_values("--ts 1,2;--ts 3,4") == ["--ts 1,2", "--ts 3,4"]

    def test_mixed(self):
        assert expand_values("99,20") == ["99", "20"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            expand_values(",,")


class TestDimensions:
    def test_template(self):
        d = Dimension(name="ngl", args="--n-gpu-layers {v}", values="99,20")
        out = d.expand()
        assert [label for label, *_ in out] == ["ngl=99", "ngl=20"]
        assert out[0][1] == ["--n-gpu-layers", "99"]

    def test_empty_template_uses_name_as_flag(self):
        d = Dimension(name="numa", args="", values="distribute,isolate,numactl")
        out = d.expand()
        assert out[0][1] == ["--numa", "distribute"]
        assert out[2][1] == ["--numa", "numactl"]

    def test_empty_template_negative_values_stay_flagged(self):
        d = Dimension(name="ngl", args="", values="-1,99")
        out = d.expand()
        assert out[0][1] == ["--ngl", "-1"]

    def test_empty_template_values_that_are_flags_pass_through(self):
        d = Dimension(name="x", args="", values="--a 1;--b 2")
        out = d.expand()
        assert out[0][1] == ["--a", "1"]

    def test_raw_fragments(self):
        d = Dimension(name="fa", args="{v}", values="--flash-attn on;--flash-attn off")
        out = d.expand()
        assert out[1][1] == ["--flash-attn", "off"]

    def test_missing_placeholder_raises(self):
        d = Dimension(name="x", args="--foo", values="1,2")
        with pytest.raises(ValueError):
            d.expand()

    def test_env_dimension_produces_env_not_args(self):
        d = Dimension(name="sdl", type="env", values="SDL_VIDEODRIVER=x11;SDL_VIDEODRIVER=headless")
        out = d.expand()
        assert [argv for _, argv, _ in out] == [[], []]
        assert out[0][2] == {"SDL_VIDEODRIVER": "x11"}
        assert out[1][2] == {"SDL_VIDEODRIVER": "headless"}

    def test_env_dimension_value_with_comma(self):
        d = Dimension(name="cuda", type="env", values="CUDA_VISIBLE_DEVICES=0,1;CUDA_VISIBLE_DEVICES=2")
        out = d.expand()
        assert out[0][2] == {"CUDA_VISIBLE_DEVICES": "0,1"}
        assert out[1][2] == {"CUDA_VISIBLE_DEVICES": "2"}

    def test_env_dimension_bare_key_means_one(self):
        d = Dimension(name="x", type="env", values="MY_FLAG")
        assert d.expand()[0][2] == {"MY_FLAG": "1"}


class TestPlanning:
    def spec(self, dims, base=""):
        return SweepSpec(
            name="t", engine="e", model="m.gguf", base_args=base,
            workloads=[Workload("pp", 512, 0)],
            dimensions=dims,
        )

    def test_cross_product(self):
        dims = [
            Dimension("a", "--a {v}", "1,2"),
            Dimension("b", "--b {v}", "x,y,z"),
        ]
        variants = plan_variants(self.spec(dims))
        assert len(variants) == 6
        assert all(len(v.args) == 4 for v in variants)

    def test_no_dimensions(self):
        variants = plan_variants(self.spec([], base="--verbose"))
        assert len(variants) == 1
        assert variants[0].args == ["--verbose"]

    def test_base_args_prefix(self):
        dims = [Dimension("a", "--a {v}", "1,2")]
        variants = plan_variants(self.spec(dims, base="--host 127.0.0.1"))
        assert variants[0].args[:2] == ["--host", "127.0.0.1"]

    def test_dedup(self):
        dims = [Dimension("a", "--a {v}", "1,1,1")]
        assert len(plan_variants(self.spec(dims))) == 1

    def test_labels(self):
        dims = [Dimension("a", "--a {v}", "1,2")]
        assert plan_variants(self.spec(dims))[1].label == "a=2"

    def test_multi_model_cross_product(self):
        spec = SweepSpec(
            name="t", engine="e",
            models=["/m/a-Q4.gguf", "/m/b-Q8.gguf"],
            workloads=[Workload("pp", 512, 0)],
            dimensions=[Dimension("flash-attn", "", "on,off")],
        )
        vs = plan_variants(spec)
        assert len(vs) == 4
        labels = sorted(v.label for v in vs)
        assert "a-Q4.gguf flash-attn=on" in labels
        assert "b-Q8.gguf flash-attn=off" in labels
        # each variant carries its own model
        assert {v.model for v in vs} == {"/m/a-Q4.gguf", "/m/b-Q8.gguf"}
        # smart dimension emitted the flag from the name
        fa_args = [v.args for v in vs if v.label == "a-Q4.gguf flash-attn=on"][0]
        assert fa_args[-2:] == ["--flash-attn", "on"]

    def test_single_model_label_omits_model(self):
        spec = SweepSpec(name="t", engine="e", models=["/m/solo.gguf"],
                         workloads=[Workload("pp", 512, 0)],
                         dimensions=[Dimension("a", "--a {v}", "1")])
        assert plan_variants(spec)[0].label == "a=1"

    def test_env_dimension_planned_into_variant_env(self):
        spec = SweepSpec(
            name="t", engine="e", model="m.gguf",
            base_env={"BASE": "1"},
            workloads=[Workload("pp", 512, 0)],
            dimensions=[
                Dimension("fa", "", "on,off"),
                Dimension("cuda", type="env", values="CUDA_VISIBLE_DEVICES=0;CUDA_VISIBLE_DEVICES=1"),
            ],
        )
        vs = plan_variants(spec)
        assert len(vs) == 4
        # arg dim -> command line, env dim -> env (not on the command line)
        v0 = vs[0]
        assert v0.args[-2:] == ["--fa", "on"]
        assert v0.env["CUDA_VISIBLE_DEVICES"] == "0"
        assert all("CUDA" not in " ".join(v0.args) for _ in [0])
        # base_env is carried on every variant separately from swept env
        assert all("BASE" in v.env for v in vs) is False  # base_env lives on the spec, not variant.env
        # distinct swept env values produced distinct variants
        assert {v.env.get("CUDA_VISIBLE_DEVICES") for v in vs} == {"0", "1"}

    def test_base_env_is_on_spec(self):
        spec = SweepSpec(name="t", engine="e", model="m.gguf",
                         base_env={"A": "1", "B": "2"},
                         workloads=[Workload("pp", 512, 0)])
        vs = plan_variants(spec)
        assert vs[0].env == {}  # nothing swept, so per-variant env is empty
        assert spec.base_env == {"A": "1", "B": "2"}


class TestFromDict:
    def test_parses_base_env_and_dim_type(self):
        spec = SweepSpec.from_dict({
            "name": "t", "engine": "e", "model": "m.gguf",
            "base_env": {"A": "1", "B": "2"},
            "dimensions": [
                {"name": "fa", "args": "", "values": "on,off", "type": "arg"},
                {"name": "cuda", "values": "CUDA_VISIBLE_DEVICES=0;=1", "type": "env"},
            ],
        })
        assert spec.base_env == {"A": "1", "B": "2"}
        assert spec.dimensions[0].type == "arg"
        assert spec.dimensions[1].type == "env"
        vs = plan_variants(spec)
        assert vs[0].env == {"CUDA_VISIBLE_DEVICES": "0"}

    def test_bad_dim_type_falls_back_to_arg(self):
        spec = SweepSpec.from_dict({
            "name": "t", "engine": "e", "model": "m.gguf",
            "dimensions": [{"name": "x", "values": "1", "type": "nonsense"}],
        })
        assert spec.dimensions[0].type == "arg"


class TestWorkload:
    def test_labels(self):
        assert Workload("pp", 512, 0).label == "pp512"
        assert Workload("tg", 0, 128).label == "tg128"
        assert Workload("pg", 512, 128).label == "pg512,128"

    def test_pp_still_generates_one(self):
        assert Workload("pp", 512, 0).max_tokens == 1

    def test_validation(self):
        with pytest.raises(ValueError):
            Workload("bogus", 1, 1)
        with pytest.raises(ValueError):
            Workload("tg", 5, 0)


class TestMetrics:
    def test_sample_stddev(self):
        import statistics
        vals = [2, 4, 4, 4, 5, 5, 7, 9]
        assert math.isclose(stdev(vals), statistics.stdev(vals), rel_tol=1e-12)
        assert math.isclose(mean(vals), statistics.mean(vals), rel_tol=1e-12)

    def test_single_value_std_zero(self):
        assert stdev([5]) == 0.0

    def test_summarize_skips_none(self):
        s = summarize([1.0, None, 3.0])
        assert (s.n, s.mean) == (2, 2.0)

    def test_percentile(self):
        vals = list(range(1, 101))
        assert percentile(vals, 50) == pytest.approx(50.5)
        assert percentile(vals, 0) == 1
        assert percentile(vals, 100) == 100
