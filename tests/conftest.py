import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9009",
        help="Base URL of the running A2A agent under test.",
    )


@pytest.fixture
def agent(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--agent-url"))
