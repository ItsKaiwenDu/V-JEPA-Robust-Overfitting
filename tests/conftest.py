"""CPU integration fixtures require an explicit, pinned upstream checkout."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--upstream", default=str(ROOT / "external/vjepa2"),
                     help="Local Meta V-JEPA 2 checkout at the configured pin.")


@pytest.fixture(scope="session")
def upstream(request):
    import torch
    from models.vjepa2 import use_upstream

    torch.set_num_threads(2)
    return use_upstream(request.config.getoption("--upstream"))
