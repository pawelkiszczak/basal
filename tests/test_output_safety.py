"""CLI options, including Make ARGS overrides, must not turn a dataset into its output."""
import sys

import pytest

from basal import bench, loadtest, run


@pytest.mark.parametrize("module,input_flag,output_flag", [
    (run, "--input", "--output"), (bench, "--questions", "--out"), (loadtest, "--questions", "--out"),
])
@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
@pytest.mark.parametrize("override_input", [False, True])
def test_effective_output_cannot_overwrite_input(tmp_path, monkeypatch, module, input_flag, output_flag, alias,
                                               override_input):
    source = tmp_path / "source.jsonl"
    original = '{"state":"keep this dataset","question":"q","options":["a","b"]}\n'
    source.write_text(original)
    output = source
    if alias != "same":
        output = tmp_path / "alias.jsonl"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    def no_work(*_, **__):
        pytest.fail("must reject the output before loading a model or contacting a server")

    monkeypatch.setattr(bench, "resolve", no_work)
    monkeypatch.setattr(run.httpx, "AsyncClient", no_work)
    # Make appends ARGS after these explicit input/output flags; argparse takes the last output.
    checked = tmp_path / "checked.jsonl"
    checked.write_text(original)
    argv = [module.__name__, input_flag, str(checked if override_input else source), output_flag,
            str(tmp_path / "safe.jsonl")]
    if override_input:
        argv += [input_flag, str(source)]
    monkeypatch.setattr(sys, "argv", argv + [output_flag, str(output)])
    with pytest.raises(ValueError, match="OUTPUT must not be an INPUT file"):
        module.main()
    assert source.read_text() == original
    assert checked.read_text() == original


def test_a_different_output_is_allowed(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    run.check_output([source], tmp_path / "answers.jsonl")
