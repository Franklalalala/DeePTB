"""Explicit offline H0 commands; importing this dispatcher loads no backend."""
import argparse
from importlib import import_module
import sys

_COMMANDS = {
    "reconstruct": "dptb.priors.h0.cli.reconstruct",
    "precompile": "dptb.priors.h0.precompile",
    "prepare-tables": "dptb.priors.h0.cli.prepare_tables",
    "compact-tables": "dptb.priors.h0.cli.compact_tables",
    "accept": "dptb.priors.h0.cli.acceptance",
}


def main(argv=None):
    """Dispatch an explicitly requested offline command, including its help."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Offline physical H0 prior preparation and reconstruction")
    parser.add_argument("command", choices=tuple(_COMMANDS))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    result = import_module(_COMMANDS[args.command]).main(argv[1:])
    return 0 if result is None else result
