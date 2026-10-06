import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu (needs a GPU node)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
