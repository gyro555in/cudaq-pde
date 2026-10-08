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

    parser.addoption(
        "--shard",
        default=None,
        metavar="K/N",
        help="run only shard K of N (1-based): tests sorted by node id, dealt round "
        "robin; keeps each login-node invocation under the 300 s CPU cap",
    )


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard expects K/N, got {spec!r}") from None
    if not 1 <= k <= n:
        raise pytest.UsageError(f"--shard K/N needs 1 <= K <= N, got {spec!r}")
    return k, n


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

    spec = config.getoption("--shard")
    if spec:
        k, n = parse_shard(spec)
        ordered = sorted(items, key=lambda it: it.nodeid)
        keep = {id(it) for i, it in enumerate(ordered) if i % n == k - 1}
        dropped = [it for it in items if id(it) not in keep]
        items[:] = [it for it in items if id(it) in keep]
        config.hook.pytest_deselected(items=dropped)
