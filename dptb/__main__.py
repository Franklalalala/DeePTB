"""Command-line entry point for the maintained DeePTB workflows."""

from dptb.entrypoints.main import main as entry_main


def main() -> None:
    entry_main()


if __name__ == "__main__":
    main()
