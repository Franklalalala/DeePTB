from dptb.entrypoints.main import main as entry_main
from dptb.entrypoints.run import run
from dptb.entrypoints.test import _test as test
from dptb.entrypoints.train import train

__all__ = ["train", "run", "test", "entry_main"]
