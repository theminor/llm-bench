"""Sweep specification: parsing, value expansion, variant planning.

Mirrors llama-bench's semantics where useful:
  - values given as comma separated lists
  - integer ranges as 'first-last', 'first-last+step', 'first-last*mult'
Unlike llama-bench, every value is an *opaque argument fragment* so that any
engine flag (or group of flags) can be swept.
"""
from __future__ import annotations

import itertools
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

_RANGE_RE = re.compile(
    r"^(-?\d+)(?:-(\d+)(?:([+*])(\d+))?)?$"
)


def parse_int_range(s: str) -> list[int]:
    """Parse a single range entry such as '512', '1-8+2', '1-32*2'."""
    m = _RANGE_RE.match(s.strip())
    if not m:
        raise ValueError(f"invalid range entry: {s!r}")
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) is not None else first
    op = m.group(3) or "+"
    step = int(m.group(4)) if m.group(4) is not None else 1
    if op == "*" and step <= 1:
        raise ValueError(f"invalid multiplier: {step}")
    out: list[int] = []
    i = first
    while True:
        if i > last:
            break
        out.append(i)
        i = i * step if op == "*" else i + step
    if not out:
        raise ValueError(f"empty range: {s!r}")
    return out


def expand_values(values: str) -> list[str]:
    """Expand a values spec into concrete value strings.

    If the spec contains ';' it is split on ';' only (so individual values
    may contain commas, e.g. '--ts 1,2;--ts 3,4'); otherwise it is split on
    commas. Each entry is either an integer range (expanded, llama-bench
    style) or a literal string (kept verbatim).
    """
    sep = ";" if ";" in values else ","
    entries = [e for e in values.split(sep) if e.strip() != ""]
    if not entries:
        raise ValueError("no values given")
    out: list[str] = []
    for entry in entries:
        entry = entry.strip()
        if _RANGE_RE.match(entry):
            out.extend(str(v) for v in parse_int_range(entry))
        else:
            out.append(entry)
    return out


@dataclass
class Dimension:
    """One swept parameter.

    ``args`` is an argument template; ``{v}`` is replaced by each value.
    Template modes:
      * empty          -> smart: emits ``--<name> <value>``; a value that is
                          already a flag (starts with ``--``) is emitted as-is
      * ``{v}``        -> raw: emits exactly the value (whole flag groups)
      * anything with ``{v}`` -> the value is substituted into the template
    """

    name: str
    args: str = ""
    values: str = ""

    def expand(self) -> list[tuple[str, list[str]]]:
        """Return list of (display_label, argv_fragments)."""
        result: list[tuple[str, list[str]]] = []
        for v in expand_values(self.values):
            if self.args.strip() == "":
                # Smart default: the dimension name is the flag, unless the
                # value is already a full flag fragment.
                fragment = v if v.startswith("--") else f"--{self.name} {v}"
            elif "{v}" in self.args:
                fragment = self.args.replace("{v}", v)
            else:
                raise ValueError(
                    f"dimension {self.name!r}: args template must contain {{v}} "
                    "unless left empty (flag = dimension name)"
                )
            argv = shlex.split(fragment) if fragment.strip() else []
            result.append((f"{self.name}={v}", argv))
        return result


@dataclass
class Workload:
    """A request shape, analogous to llama-bench's pp / tg / pg tests."""

    kind: str  # "pp" | "tg" | "pg"
    n_prompt: int
    n_gen: int

    def __post_init__(self) -> None:
        if self.kind not in ("pp", "tg", "pg"):
            raise ValueError(f"unknown workload kind {self.kind!r}")
        if self.kind == "pp" and self.n_prompt <= 0:
            raise ValueError("pp workload needs n_prompt > 0")
        if self.kind == "tg" and self.n_gen <= 0:
            raise ValueError("tg workload needs n_gen > 0")
        if self.kind == "pg" and (self.n_prompt <= 0 or self.n_gen <= 0):
            raise ValueError("pg workload needs n_prompt > 0 and n_gen > 0")

    @property
    def label(self) -> str:
        if self.kind == "pp":
            return f"pp{self.n_prompt}"
        if self.kind == "tg":
            return f"tg{self.n_gen}"
        return f"pg{self.n_prompt},{self.n_gen}"

    @property
    def max_tokens(self) -> int:
        # pp still needs one generated token over HTTP; measure prefill via TTFT.
        return max(1, self.n_gen)


@dataclass
class SweepSpec:
    name: str
    engine: str
    model: str = ""
    base_args: str = ""            # applied to every variant (shlex string)
    workloads: list[Workload] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    repetitions: int = 3
    warmup: bool = True
    cooldown_s: float = 2.0
    port: int = 0                  # 0 = auto-assign
    startup_timeout_s: float = 300.0
    request_timeout_s: float = 600.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SweepSpec":
        return SweepSpec(
            name=str(d.get("name") or "sweep"),
            engine=str(d["engine"]),
            model=str(d.get("model", "")),
            base_args=str(d.get("base_args", "")),
            workloads=[Workload(**w) for w in d.get("workloads", [])],
            dimensions=[Dimension(**dim) for dim in d.get("dimensions", [])],
            repetitions=int(d.get("repetitions", 3)),
            warmup=bool(d.get("warmup", True)),
            cooldown_s=float(d.get("cooldown_s", 2.0)),
            port=int(d.get("port", 0)),
            startup_timeout_s=float(d.get("startup_timeout_s", 300.0)),
            request_timeout_s=float(d.get("request_timeout_s", 600.0)),
        )


@dataclass
class Variant:
    labels: dict[str, str]
    args: list[str]
    label: str

    @property
    def key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(self.labels.items()))


def plan_variants(spec: SweepSpec) -> list[Variant]:
    """Cross-product of all dimensions (llama-bench style), deduplicated."""
    expanded = [dim.expand() for dim in spec.dimensions]
    base = shlex.split(spec.base_args) if spec.base_args.strip() else []
    seen: set[str] = set()
    variants: list[Variant] = []
    if not expanded:
        variants.append(Variant(labels={}, args=base, label="(baseline)"))
        return variants
    for combo in itertools.product(*expanded):
        labels = {name: value for name, value in ((c[0].split("=", 1)[0], c[0].split("=", 1)[1]) for c in combo)}
        args = list(base)
        for _, argv in combo:
            args.extend(argv)
        v = Variant(labels=labels, args=args, label=" ".join(combo_c[0] for combo_c in combo))
        if v.key in seen:
            continue
        seen.add(v.key)
        variants.append(v)
    return variants
