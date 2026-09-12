"""Geometry/physics input and a separate, post-construction ABACUS CSR oracle."""
from pathlib import Path
import re
import numpy as np
from scipy.sparse import csr_matrix
from h0rebuild.models import Atom, BlockKey, SpeciesData, Structure
from h0rebuild.orb import read_abacus_orb
from h0rebuild.upf import read_upf

RY_TO_EV = 13.605693122994


def load_case(path, *, prepared_species=None):
    path = Path(path)
    if prepared_species is None:
        import os, json
        store = os.environ.get('H0_OFFLINE_TABLE_DIR')
        if store:
            from h0rebuild.offline import load_species
            from h0rebuild.precompiled import sha256
            catalog = json.loads((Path(store)/'catalog.json').read_text())
            if path.resolve().parent != Path(catalog['raw']).resolve():
                raise ValueError('Case is outside the prepared immutable cohort; prepare its species explicitly')
            entry = catalog['cases'][path.name]
            if sha256(path/'STRU') != entry['STRU_sha256']: raise ValueError('STRU changed since offline preparation')
            prepared_species = {s:load_species(store,i) for s,i in entry['species'].items()}
    lines = [x.split('#')[0].strip() for x in (path/'STRU').read_text().splitlines()]
    lines = [x for x in lines if x]
    ia, io = lines.index('ATOMIC_SPECIES'), lines.index('NUMERICAL_ORBITAL')
    il, iv, ip = [lines.index(x) for x in ['LATTICE_CONSTANT','LATTICE_VECTORS','ATOMIC_POSITIONS']]
    spec = [x.split() for x in lines[ia+1:io]]
    orbs = lines[io+1:il]
    lat0 = float(lines[il+1])
    cell = np.array([[float(v) for v in x.split()] for x in lines[iv+1:iv+4]]) * lat0
    mode = lines[ip+1].lower()
    if mode not in ('direct','cartesian'):
        raise ValueError('Unsupported coordinate convention: '+mode)
    pos = ip+2
    atoms = []
    moments = []
    for symbol, _, _ in spec:
        assert lines[pos] == symbol
        species_moment = float(lines[pos+1])
        n = int(lines[pos+2])
        for row in lines[pos+3:pos+3+n]:
            words = row.split()
            xyz = np.array([float(v) for v in words[:3]])
            moment = species_moment
            if 'mag' in words:
                start = words.index('mag') + 1
                values = []
                for word in words[start:]:
                    try: values.append(float(word))
                    except ValueError: break
                if len(values) == 1: moment = values[0]
                elif len(values) == 3 and values[0] == values[1] == 0: moment = values[2]
                else: raise ValueError('Only collinear z initial magnetization is supported')
            if 'angle1' in words or 'angle2' in words: raise ValueError('Magnetization angles require an explicit vector-field implementation')
            moments.append(moment)
            frac = xyz if mode == 'direct' else (xyz * lat0) @ np.linalg.inv(cell)
            atoms.append(Atom(symbol, frac))
        pos += 3+n
    sd = (dict(prepared_species) if prepared_species is not None else
          {entry[0]: SpeciesData(read_abacus_orb(path/'PP_ORB'/orb), read_upf(path/'PP_ORB'/entry[2]))
           for entry, orb in zip(spec, orbs)})
    if set(sd) != {entry[0] for entry in spec}: raise ValueError('Prepared species do not match structure')
    raw = (path/'OUT.ABACUS'/'INPUT').read_text()
    inp = {w[0]: w[1] for line in raw.splitlines() if len(w:=line.split('#')[0].split())>=2}
    spin=int(inp['nspin'])
    assert spin in (1,4) and int(inp['lspinorb'])==(1 if spin==4 else 0)
    assert inp['init_chg']=='atomic'
    assert all('PBE' in d.upf.functional.upper() for d in sd.values())
    log = (path/'OUT.ABACUS'/'running_scf.log').read_text()
    # ABACUS may move individual input atoms by whole lattice vectors before
    # writing CSR blocks. Keep the calculation geometry, but request that the
    # core H0 builder emits all matrices in the documented output home cells.
    from h0rebuild.cell_gauge import shifts_between_coordinates
    logged = [line.split() for line in log.splitlines() if line.startswith('tauc_')]
    if len(logged) != len(atoms):
        raise ValueError('Expected one ABACUS output Cartesian position per atom')
    for row, atom in zip(logged, atoms):
        if re.sub(r'\d+$', '', row[0].removeprefix('tauc_')) != atom.species:
            raise ValueError('ABACUS output atom order does not match STRU')
    logged_cart = np.asarray([[float(v) for v in row[1:4]] for row in logged]) * lat0
    output_shifts = shifts_between_coordinates(cell, [a.frac for a in atoms], logged_cart)
    grids = re.findall(r'^\s*fft grid for charge/potential\s*=\s*\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]',log,re.M|re.I)
    assert len(set(grids))==1
    opts = {'ecutrho_ry':float(inp['ecutrho']), 'fft_shape':tuple(map(int,grids[0])), 'xc':'PBE',
            'pseudo_rcut_bohr':float(inp['pseudo_rcut']),
            'nspin':spin, 'include_nlcc':True, 'total_electrons':sum(sd[a.species].upf.z_valence for a in atoms),
            'output_atom_cell_shifts':output_shifts}
    if spin == 4 and any(m != 0 for m in moments): opts['initial_moments_z'] = moments
    contract = {k: inp.get(k) for k in ['nspin','lspinorb','ecutwfc','ecutrho','init_chg','dft_functional','nelec']}
    contract.update({'grid':opts['fft_shape'], 'species':[a.species for a in atoms], 'source_path':str(path),
                     'initial_moments_z':moments,'reader_revision':'atomic-mag/v2',
                     'cell_bohr':cell.tolist(),'fractional_coordinates':[a.frac.tolist() for a in atoms],
                     'output_atom_cell_shifts':output_shifts.tolist(),
                     'output_coordinate_contract':'ABACUS logged home cells, core H0/S/component/metadata rebasing'})
    return Structure(cell,atoms), sd, opts, contract


def read_csr(path, counts, nspin=1):
    """Raw ABACUS order and Ry units; no phase fit, shift, or convention search."""
    blocks = {}
    offsets = np.r_[0,np.cumsum(counts)]
    with Path(path).open() as f:
        step=f.readline().strip()
        dim=int(f.readline().split(':')[-1])
        nr=int(f.readline().split(':')[-1])
        assert dim==offsets[-1], (dim,offsets[-1])
        for _ in range(nr):
            header=[int(x) for x in f.readline().split()]
            assert len(header)==4
            *r,nnz=header
            if nnz:
                line=f.readline()
                data=(np.asarray([complex(float(a),float(b)) for a,b in re.findall(r'\(([^,]+),([^\)]+)\)',line)])
                      if '(' in line else np.fromstring(line,sep=' '))
                indices=np.fromstring(f.readline(),sep=' ',dtype=np.int64)
                indptr=np.fromstring(f.readline(),sep=' ',dtype=np.int64)
                assert len(data)==len(indices)==nnz and len(indptr)==dim+1
                matrix=csr_matrix((data,indices,indptr),shape=(dim,dim)).toarray()
            else:matrix=np.zeros((dim,dim))
            for i in range(len(counts)):
                for j in range(len(counts)):
                    b=matrix[offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]
                    if nspin==4:
                        # ABACUS CSR interleaves spin within each spatial AO;
                        # H0Flash uses spin-block-major inside each atom block.
                        pi=np.r_[np.arange(0,counts[i],2),np.arange(1,counts[i],2)]
                        pj=np.r_[np.arange(0,counts[j],2),np.arange(1,counts[j],2)]
                        b=b[np.ix_(pi,pj)]
                    if np.any(b):blocks[BlockKey(i,j,tuple(r))]=b
        assert not f.read().strip(), 'Unexpected trailing CSR data'
    return blocks, {'step':step,'matrix_dimension':dim,'translation_count':nr,'block_count':len(blocks)}


def compare_blocks(pred, ref, counts, scale=1.):
    metrics={}
    for group in ['total','onsite','hopping']:
        errors=[]; references=[]; worst=None
        for key in set(pred)|set(ref):
            onsite=key.i==key.j and key.R==(0,0,0)
            if (group=='onsite' and not onsite) or (group=='hopping' and onsite):continue
            shape=(counts[key.i],counts[key.j])
            truth=np.asarray(ref.get(key,np.zeros(shape)))*scale
            guess=np.asarray(pred.get(key,np.zeros(shape)))*scale
            assert truth.shape==guess.shape==shape
            delta=guess-truth
            idx=np.unravel_index(np.argmax(np.abs(delta)),delta.shape)
            val=float(abs(delta[idx]))
            if worst is None or val>worst['abs_error']:
                worst={'key':key.as_tuple(),'row':int(idx[0]),'column':int(idx[1]),'abs_error':val,
                       'reference_real':float(truth[idx].real),'reference_imag':float(truth[idx].imag),'prediction_real':float(guess[idx].real),'prediction_imag':float(guess[idx].imag)}
            errors.append(delta.ravel());references.append(truth.ravel())
        e=np.concatenate(errors);r=np.concatenate(references)
        active=np.abs(r)>1e-6*scale
        metrics[group]={'elements':len(e),'reference_nonzero_elements':int(np.count_nonzero(active)),
                        'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.mean(np.abs(e)**2))),
                        'max_abs':float(np.max(np.abs(e))),
                        'mae_on_reference_nonzero':float(np.abs(e[active]).mean()),
                        'reference_rms':float(np.sqrt(np.mean(np.abs(r)**2))),'worst':worst}
    metrics['support']={'predicted_blocks':len(pred),'reference_blocks':len(ref),
                        'missing_reference_blocks':len(set(ref)-set(pred)),
                        'extra_predicted_blocks':len(set(pred)-set(ref))}
    return metrics
