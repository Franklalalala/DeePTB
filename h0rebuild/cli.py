"""Legacy reconstruction command; the implementation lives in DeePTB."""
from dptb.priors.h0.cli.reconstruct import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
