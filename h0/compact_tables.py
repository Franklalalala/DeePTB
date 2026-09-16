"""Explicit, read-only-source conversion to a new shared radial table directory."""
import argparse
import json
from h0rebuild.shared_radial_store import convert_store

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='existing verified offline v2 table directory')
    parser.add_argument('destination', help='new directory; must not exist')
    args = parser.parse_args()
    result = convert_store(args.source, args.destination)
    print(json.dumps({k:v for k,v in result.items() if k!='tables'}, indent=2))
