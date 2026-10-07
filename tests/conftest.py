import pytest

pytest_plugins = ["pytester"]  # for tests/test_markers.py


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu (needs a GPU node)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (each takes more than about 1 s)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skips = {
        "gpu": ("--run-gpu", "needs --run-gpu"),
        "slow": ("--run-slow", "slow test, needs --run-slow"),
    }
    for keyword, (option, reason) in skips.items():
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)
