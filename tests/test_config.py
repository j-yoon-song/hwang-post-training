from synth_parallel.config import load_config


def test_metricx_batch_size_forced_to_one(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "metricx:\n"
        "  batch_size: 128\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.metricx.batch_size == 1
