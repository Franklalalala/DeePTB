"""Prior table builders behind one entry point (``dptb prior build``): family name -> builder module.

A builder module exposes
    build(keys, out_dir, *, reference, **options) -> {key: seconds}   write tables of these keys into out_dir
    species_of(reference) -> [symbol]                                  species of a reference table set
    present(out_dir) -> {key}                                          keys already built in out_dir
    ORDERED                                                            'A|B' and 'B|A' are distinct tables
    PER_SPECIES (optional)                                             keys are species symbols, not pairs
Tables are written only into extension roots, ``<out>/<family>`` with their own manifests (layout of the deployed
table roots, so the readers can stack them after the deployed ones); deployed tables are never written
(``dptb.prior.common.store`` refuses to replace files). ``build_missing`` builds only the keys an extension root lacks;
the readers' on-the-fly switch calls it (off by default).
"""
import argparse
import json
import time
from importlib import import_module
from pathlib import Path

FAMILIES = {
    'h0': 'dptb.prior.h0.build',
    'overlap': 'dptb.prior.gf.build_overlap',
    'p2': 'dptb.prior.gf.build_p2',
    'soc': 'dptb.prior.gf.build_soc',
    'moments': 'dptb.prior.gf.build_moments',
}


def _manifest(root, names=('manifest.json',)):
    for n in names:
        if (Path(root) / n).exists():
            return json.loads((Path(root) / n).read_text())
    return {}


def _files(keys, pporb, species_files):
    """ORB/UPF of every species named in ``keys`` ('A|B' or 'kind:A|B')."""
    from dptb.prior.gf.build_p23 import species_files as resolve
    return resolve(sorted({s for k in keys for s in k.split(':', 1)[-1].split('|')}), pporb, files=species_files)


class P23:
    """P23 factor tables 'V|A' (``dptb.prior.gf.build_p23``)."""
    ORDERED = True

    @staticmethod
    def species_of(reference):
        return sorted(_manifest(reference).get('species', {}))

    @staticmethod
    def present(out_dir):
        return set(_manifest(out_dir).get('factor_tables', {}))

    @staticmethod
    def build(keys, out_dir, *, reference, pporb, species_files=None, device='cuda', log=print):
        from dptb.prior.gf import build_p23
        rows = {s: r for s, r in _manifest(reference).get('species', {}).items() if 'orbital_sha256' in r} or None
        return build_p23.build(keys, out_dir, pporb, device, reference=rows, species_files_map=species_files, log=log)[1]


class PairXC:
    """Pair-XC tables 'A|B' with A <= B, homo pairs 'A|A' (``dptb.prior.gf.build_pairxc``)."""
    ORDERED = True
    NAMES = ('manifest.json', 'manifest_merged.json')

    @staticmethod
    def element_keys(element, species):
        return ['|'.join(sorted((element, s))) for s in species]

    @staticmethod
    def species_of(reference):
        m = _manifest(reference, PairXC.NAMES) or _manifest(Path(reference) / 'pairxc', PairXC.NAMES)
        return sorted({s for k in m.get('pairs', {}) for s in k.split('|')})

    @staticmethod
    def present(out_dir):
        return set(_manifest(out_dir).get('pairs', {}))

    @staticmethod
    def build(keys, out_dir, *, reference, pporb, species_files=None, device='cuda', log=print):
        from dptb.prior.gf import build_pairxc
        res = build_pairxc.build(keys, out_dir, species_files=_files(keys, pporb, species_files), device=device)
        return {k: sec for k, _, sec in res}


class EnvXC:
    """Environment-XC tables 'kind:L|R' and the species projectors (``dptb.prior.gf.build_envxc``)."""
    ORDERED = True

    @staticmethod
    def element_keys(element, species):
        from dptb.prior.gf.build_envxc import items_for
        return [f'{k}:{a}|{b}' for k, a, b in items_for([element], species)]

    @staticmethod
    def species_of(reference):
        return sorted(_manifest(reference).get('species', {}))

    @staticmethod
    def present(out_dir):
        m = _manifest(out_dir)
        return {f'{kind}:{key}' for kind, rows in m.get('tables', {}).items() for key in rows}

    @staticmethod
    def build(keys, out_dir, *, reference, pporb, species_files=None, device='cuda', log=print):
        from dptb.prior.gf import build_envxc
        res = build_envxc.build(keys, out_dir, species_files=_files(keys, pporb, species_files), device=device)
        return {k: sec for k, _, sec in res}


FAMILIES.update(p23=P23, pairxc=PairXC, envxc=EnvXC)


def register(name, module):
    """Add a family: a module path or an object exposing the builder interface above."""
    FAMILIES[name] = module


def family(name):
    if name not in FAMILIES:
        raise KeyError(f'unknown table family {name!r}; known: {sorted(FAMILIES)}')
    f = FAMILIES[name]
    return import_module(f) if isinstance(f, str) else f


def reference_root(name, reference):
    """A family root: ``reference`` itself or its ``<name>`` subdirectory (a tables root)."""
    ref = Path(reference)
    return ref / name if (ref / name / 'manifest.json').exists() else ref


def keys_for(name, *, pairs=(), element=None, reference=None, species=None):
    """Table keys of explicit pairs ('A|B'; species symbols for per-species families) plus, for ``element``, every
    table between it and each species of the reference set (and itself)."""
    mod = family(name)
    per_species = getattr(mod, 'PER_SPECIES', False)
    out = []
    for p in pairs:
        parts = p.split('|') if isinstance(p, str) else list(p)
        out.append(parts[0] if per_species else
                   '|'.join(parts if mod.ORDERED else sorted(set(parts))))
    if element and hasattr(mod, 'element_keys'):
        out += mod.element_keys(element, sorted(set(species or mod.species_of(reference)) | {element}))
    elif element:
        if per_species:
            out.append(element)
        else:
            others = sorted(set(species or mod.species_of(reference)) | {element})
            for s in others:
                out.append('|'.join((element, s) if mod.ORDERED else sorted({element, s})))
                if mod.ORDERED and s != element:
                    out.append(f'{s}|{element}')
    return list(dict.fromkeys(out))


def build(name, out_dir, *, reference, pairs=(), element=None, species=None, **options):
    """Build the tables of ``pairs``/``element`` for one family into ``<out_dir>/<name>``."""
    ref = reference_root(name, reference)
    keys = keys_for(name, pairs=pairs, element=element, reference=ref, species=species)
    return family(name).build(keys, Path(out_dir) / name, reference=ref, **options)


def build_missing(name, pairs, out_dir, *, reference, **options):
    """Build only the keys of ``pairs`` that ``<out_dir>/<name>`` does not hold yet."""
    mod = family(name)
    missing = [k for k in keys_for(name, pairs=pairs) if k not in mod.present(Path(out_dir) / name)]
    if not missing:
        return {}
    return mod.build(missing, Path(out_dir) / name, reference=reference_root(name, reference), **options)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='dptb prior', description='Prior table tools.')
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='Build prior tables on the GPU into an extension directory.')
    b.add_argument('families', nargs='+', help=f'table families: {", ".join(sorted(FAMILIES))}')
    b.add_argument('--out', required=True, help='extension root; each family goes to <out>/<family>')
    b.add_argument('--reference', required=True,
                   help='reference tables root (or family root): physical contract, settings and known species')
    b.add_argument('--h0-reference', help='reference H0 pair bank (default: --reference)')
    b.add_argument('--pairs', default='', help="comma-separated keys, e.g. 'H|O,O|H' (species for per-species families)")
    b.add_argument('--element', help='build every table between this element and the reference species')
    b.add_argument('--pporb', action='append', default=[], help='directory with ORB/UPF files (repeatable)')
    b.add_argument('--species-files', action='append', default=[], metavar='X=ORB,UPF',
                   help='ORB and UPF files of a species the reference lacks (repeatable)')
    b.add_argument('--missing-only', action='store_true', help='skip keys the extension already holds')
    b.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    files = {}
    for item in args.species_files:
        sym, paths = item.split('=', 1)
        files[sym] = tuple(paths.split(','))
    pairs = [p for p in args.pairs.split(',') if p]
    report = {}
    for name in args.families:
        ref = args.h0_reference if name == 'h0' and args.h0_reference else args.reference
        options = dict(species_files=files or None, device=args.device)
        if name != 'h0':
            options['pporb'] = args.pporb
        t = time.perf_counter()
        if args.missing_only:
            keys = keys_for(name, pairs=pairs, element=args.element, reference=reference_root(name, ref))
            seconds = build_missing(name, keys, args.out, reference=ref, **options)
        else:
            seconds = build(name, args.out, reference=ref, pairs=pairs, element=args.element, **options)
        report[name] = dict(tables=len(seconds), device_seconds=sum(seconds.values()), wall_seconds=time.perf_counter() - t)
        print(json.dumps({name: report[name]}), flush=True)
    return 0
