"""SpecGate (generator.generator.SpecGate): option handling and the cost comparison.

CPU-only: the gate is plain Python arithmetic over two EMAs, no model, cache or device involved.
"""
from exllamav3.generator.generator import SpecGate


def test_explicit_zero_options_are_not_replaced_by_the_environment(monkeypatch):
    # `x or default` used to swallow these: alpha=0.0 (freeze the EMAs at their first sample),
    # probe_min=0 (probe after every plain step) and margin=0.0 (a probe that merely breaks even pays
    # for itself) are all meaningful settings, so only None may fall through to the environment
    monkeypatch.setenv("EXL3_SPEC_GATE_ALPHA", "0.3")
    monkeypatch.setenv("EXL3_SPEC_GATE_PROBE", "4")
    monkeypatch.setenv("EXL3_SPEC_GATE_PROBE_MAX", "32")
    monkeypatch.setenv("EXL3_SPEC_GATE_MARGIN", "1.0")
    gate = SpecGate(alpha = 0.0, probe_min = 0, probe_max = 0, margin = 0.0)
    assert gate.alpha == 0.0
    assert gate.probe_min == 0
    assert gate.probe_max == 0
    assert gate.margin == 0.0
    assert gate.probe == 0                    # starts at probe_min


def test_unset_options_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("EXL3_SPEC_GATE_ALPHA", "0.5")
    monkeypatch.setenv("EXL3_SPEC_GATE_PROBE", "8")
    monkeypatch.setenv("EXL3_SPEC_GATE_PROBE_MAX", "64")
    monkeypatch.setenv("EXL3_SPEC_GATE_MARGIN", "2.0")
    gate = SpecGate()
    assert (gate.alpha, gate.probe_min, gate.probe_max, gate.margin) == (0.5, 8, 64, 2.0)


def test_the_smoothing_factor_is_used_as_given():
    # _ema = old + alpha * (new - old): alpha 0.0 pins the estimate at its first sample, 1.0 replaces
    # it. Both are legitimate settings, which is why 0.0 must not fall through to the env default
    gate = SpecGate(alpha = 0.0)
    assert gate._ema(10.0, 4.0) == 10.0
    assert gate._ema(None, 4.0) == 4.0        # first sample is taken as is
    gate = SpecGate(alpha = 1.0)
    assert gate._ema(10.0, 4.0) == 4.0


def test_a_zero_length_step_does_not_divide_by_zero():
    # t_spec/t_plain are EMAs of perf_counter deltas; a step timed at exactly 0.0 s used to raise
    # ZeroDivisionError inside iterate()
    gate = SpecGate(margin = 0.0)
    gate.t_spec, gate.t_plain, gate.tok_round = 0.0, 1.0, 1.0
    assert gate.spec_pays() is True
    gate.t_spec, gate.t_plain, gate.tok_round = 0.0, 0.0, 1.0
    assert isinstance(gate.spec_pays(), bool)
    gate.t_spec, gate.t_plain, gate.tok_round = 0.02, 0.0, 2.0
    assert isinstance(gate.spec_pays(), bool)


def test_the_gate_switches_to_plain_and_back():
    gate = SpecGate(alpha = 1.0, margin = 1.0)      # alpha 1.0: every sample replaces the EMA
    gate.record_plain(0.01, 1)                      # one token per 10 ms: 100 tok/s
    gate.record_spec(0.05, 2)                       # two tokens per 50 ms: 40 tok/s, does not pay
    assert gate.spec_pays() is False
    assert gate.mode_spec is False                  # a regular round turned speculation off
    assert gate.probe == 4                          # only a failed *probe* changes the interval
    for _ in range(4):
        gate.record_plain(0.01, 1)
    assert gate.want_spec() is True                 # since_probe reached probe
    gate.record_spec(0.05, 2)                       # the probe fails: double the interval
    assert gate.mode_spec is False
    assert gate.probe == 8
    assert gate.since_probe == 0
    for _ in range(8):
        gate.record_plain(0.01, 1)
    assert gate.want_spec() is True
    gate.record_spec(0.02, 8)                       # 8 tokens per 20 ms: 400 tok/s, pays
    assert gate.mode_spec is True
    assert gate.probe == 4                          # reset to probe_min


def test_a_failed_probe_is_capped_by_probe_max():
    gate = SpecGate(alpha = 1.0, probe_min = 4, probe_max = 6)
    gate.record_plain(0.01, 1)                      # 100 tok/s plain
    for _ in range(6):
        for _ in range(gate.probe):
            gate.record_plain(0.01, 1)
        gate.record_spec(0.5, 1)                    # 2 tok/s: never pays
    assert gate.mode_spec is False
    assert gate.probe == 6                          # 4 -> 8 capped to 6, and stays there
