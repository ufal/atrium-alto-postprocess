import pytest

from alto_stats_create import main


def test_alto_stats_cli_help(capsys):
    with pytest.raises(SystemExit) as e:  # argparse exits 0 on --help
        main(["--help"])
    assert e.value.code == 0
    assert "input_folder" in capsys.readouterr().out


def test_alto_stats_missing_args(capsys):
    with pytest.raises(SystemExit) as e:  # missing required positional → exit 2
        main([])
    assert e.value.code == 2
    assert "required" in capsys.readouterr().err.lower()
