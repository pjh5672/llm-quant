import pytest
import yaml

from llmquant.core.parser import RunConfig, build_config

BASE = {
    "defaults": {"project": "p", "model": "m", "seq_len": 512},
    "quantization": {"attn_weight": "int4", "mlp_weight": "int8", "activation": "int8"},
    "evaluation": {"metric": "wikitext2"},
}


def write(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_yaml_sections_are_flattened(tmp_path):
    c = build_config(["--cfg", write(tmp_path, BASE)], root_dir=tmp_path)
    assert (c.project, c.model, c.seq_len, c.metric) == ("p", "m", 512, "wikitext2")
    assert c.quant.attn_weight == "int4" and c.quant.mlp_weight == "int8"


def test_cli_overrides_yaml(tmp_path):
    c = build_config(
        ["--cfg", write(tmp_path, BASE), "--mlp-weight", "int4", "--seq-len", "128"],
        root_dir=tmp_path,
    )
    assert c.quant.mlp_weight == "int4" and c.seq_len == 128


def test_missing_keys_fall_back_to_defaults(tmp_path):
    c = build_config(["--cfg", write(tmp_path, {"defaults": {"project": "p"}})], root_dir=tmp_path)
    assert c.quant.head_weight == RunConfig().quant.head_weight
    assert c.seq_len == RunConfig().seq_len


def test_config_file_is_copied_into_the_experiment_dir(tmp_path):
    c = build_config(["--cfg", write(tmp_path, BASE)], root_dir=tmp_path)
    assert c.project_dir == tmp_path / "experiments" / "p"
    assert (c.project_dir / "cfg.yaml").exists()


def test_unknown_section_and_key_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config section"):
        build_config(["--cfg", write(tmp_path, {"nope": {}})], root_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown config key"):
        build_config(["--cfg", write(tmp_path, {"defaults": {"nope": 1}})], root_dir=tmp_path)


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("quantization", "mode", "real", "only 'fake' exists"),
        ("quantization", "quant_method", "gptq", "only calibration-free"),
        ("defaults", "pack", True, "pack is not implemented"),
        ("defaults", "load_packed", "x.bin", "load_packed is not implemented"),
    ],
)
def test_unimplemented_slots_raise_instead_of_no_op(tmp_path, section, key, value, match):
    cfg = {"defaults": dict(BASE["defaults"]), "quantization": dict(BASE["quantization"])}
    cfg[section][key] = value
    with pytest.raises(NotImplementedError, match=match):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


def test_no_quantize_yields_no_modifier(tmp_path):
    cfg = {"defaults": {"project": "p"}, "quantization": {"quantize": False}}
    assert build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path).to_modifier() is None


def test_sweep_expands_to_the_cross_product(tmp_path):
    cfg = {
        "defaults": {"project": "p"},
        "sweep": {"attn_weight": ["int4", "int8"], "mlp_weight": ["int4", "int8"]},
    }
    from llmquant.core.parser import expand_sweep

    runs = expand_sweep(build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path))
    assert len(runs) == 4
    assert {(r.quant.attn_weight, r.quant.mlp_weight) for r in runs} == {
        ("int4", "int4"), ("int4", "int8"), ("int8", "int4"), ("int8", "int8")
    }
    # everything outside the grid is shared
    assert {r.project for r in runs} == {"p"} and {r.seq_len for r in runs} == {2048}


def test_sweep_collapses_runs_that_resolve_to_the_same_schemes(tmp_path):
    from llmquant.core.parser import expand_sweep

    # all weights bf16 -> the activation dtype has nothing to act on
    cfg = {"defaults": {"project": "p"}, "sweep": {"activation": ["bf16", "int8"]}}
    runs = expand_sweep(build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path))
    assert len(runs) == 1

    cfg = {
        "defaults": {"project": "p"},
        "quantization": {"mlp_weight": "int4"},
        "sweep": {"activation": ["bf16", "int8"]},
    }
    runs = expand_sweep(build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path))
    assert len(runs) == 2


def test_no_sweep_section_yields_a_single_run(tmp_path):
    from llmquant.core.parser import expand_sweep

    runs = expand_sweep(build_config(["--cfg", write(tmp_path, BASE)], root_dir=tmp_path))
    assert len(runs) == 1 and runs[0].quant.attn_weight == "int4"


@pytest.mark.parametrize(
    "sweep,match",
    [
        ({"nope": ["int4"]}, "unknown sweep key"),
        ({"attn_weight": "int4"}, "must be a non-empty list"),
        ({"attn_weight": []}, "must be a non-empty list"),
    ],
)
def test_bad_sweep_section_is_rejected(tmp_path, sweep, match):
    cfg = {"defaults": {"project": "p"}, "sweep": sweep}
    with pytest.raises(ValueError, match=match):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


def test_selection_section_is_parsed(tmp_path):
    cfg = {
        "defaults": {"project": "p"},
        "selection": {"acc_drop_limit_pct": 2.0, "bpv_weight": 5.0, "prefill_speed_weight": 3.0},
    }
    c = build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)
    assert c.selection.acc_drop_limit_pct == 2.0
    assert c.selection.bpv_weight == 5.0 and c.selection.prefill_speed_weight == 3.0
    assert c.selection.accuracy_weight == 1.0  # untouched keys keep their default


def test_unknown_selection_key_is_rejected(tmp_path):
    cfg = {"defaults": {"project": "p"}, "selection": {"nope": 1}}
    with pytest.raises(ValueError, match="unknown selection key"):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"acc_drop_limit_pct": -1}, "must be >= 0"),
        ({"bpv_weight": -1}, "must be >= 0"),
        ({"prefill_speed_weight": -1}, "must be >= 0"),
    ],
)
def test_bad_selection_values_are_rejected(tmp_path, selection, match):
    cfg = {"defaults": {"project": "p"}, "selection": selection}
    with pytest.raises(ValueError, match=match):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


def test_stage_two_options_default_and_override(tmp_path):
    c = build_config(["--cfg", write(tmp_path, BASE)], root_dir=tmp_path)
    assert c.generation is True and c.lambada_limit == 500 and c.max_new_tokens == 64
    c = build_config(
        ["--cfg", write(tmp_path, BASE), "--no-generation", "--lambada-limit", "50"],
        root_dir=tmp_path,
    )
    assert c.generation is False and c.lambada_limit == 50


def test_task_suite_defaults(tmp_path):
    c = build_config(["--cfg", write(tmp_path, BASE)], root_dir=tmp_path)
    assert c.tasks == ("arc_easy", "arc_challenge", "openbookqa")
    assert c.task_limit is None and c.ppl is True and c.latency is True


def test_tasks_and_ppl_can_be_selected_independently(tmp_path):
    cfg = {**BASE, "evaluation": {**BASE["evaluation"], "tasks": ["gsm8k"], "ppl": False}}
    c = build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)
    assert c.tasks == ("gsm8k",) and c.ppl is False


def test_unknown_task_is_rejected(tmp_path):
    cfg = {**BASE, "evaluation": {**BASE["evaluation"], "tasks": ["hellaswag"]}}
    with pytest.raises(ValueError, match="unknown task"):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


def test_measuring_nothing_is_rejected(tmp_path):
    cfg = {**BASE, "evaluation": {**BASE["evaluation"], "tasks": [], "ppl": False}}
    with pytest.raises(ValueError, match="nothing to measure"):
        build_config(["--cfg", write(tmp_path, cfg)], root_dir=tmp_path)


def test_tasks_and_latency_can_be_overridden_from_the_cli(tmp_path):
    c = build_config(
        ["--cfg", write(tmp_path, BASE), "--tasks", "gsm8k", "arc_easy",
         "--task-limit", "20", "--no-latency"],
        root_dir=tmp_path,
    )
    assert c.tasks == ("gsm8k", "arc_easy") and c.task_limit == 20 and c.latency is False
