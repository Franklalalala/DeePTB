"""Explicit, read-only-source conversion to a new shared radial table directory."""
import argparse
import json

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='existing verified offline v2 table directory')
    parser.add_argument('destination', help='new directory; must not exist')
    args = parser.parse_args(argv)
    from dptb.priors.h0.shared_radial_store import convert_store

    result = convert_store(args.source, args.destination)
    print(json.dumps({k:v for k,v in result.items() if k!='tables'}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
