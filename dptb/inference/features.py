"""Block tables <-> model features as device gathers.

The models read compact uu-real features (``nextham_uureal_mask``) and predict a compact uu-real delta. The reference
conversions are DeePTB's ``block_to_feature`` / ``feature_to_block`` on the full spinor mapper (complex doubling), with
the compact <-> full layout of every orbital pair: the compact slice of an orbital pair is the uu-real slice of the
full features, and a compact delta is written to both the uu-real and dd-real slices. Both conversions are pure
selections, so they are derived once per species set by running the reference functions on index-coded blocks and
features, and then applied as one gather per species (nodes) or species pair (edges).
"""
import copy

import numpy as np
import torch
from ase.data import atomic_numbers as Z_OF

from dptb.data import AtomicDataDict as K
from dptb.data.interfaces.ham_to_feature import block_to_feature, feature_to_block
from dptb.data.transforms import OrbitalMapper


def spinor_mapper(cidp, device):
    """Full spinor mapper (complex doubling) of a compact uu-real model mapper."""
    fidp = OrbitalMapper(basis=copy.deepcopy(cidp.basis), chemical_symbol_to_type=cidp.chemical_symbol_to_type,
                         method=cidp.method, device=device, has_soc=True, soc_complex_doubling=True,
                         nextham_uureal_mask=False)
    for m in (cidp, fidp):
        m.get_orbital_maps()
        m.get_orbpair_maps()
    return fidp


def layout_maps(cidp, fidp):
    """Index arrays of the compact <-> full layout: compact column c is full uu-real column uu[c] (= dd column dd[c])."""
    uu, dd = [], []
    for pair, csl in sorted(cidp.orbpair_maps.items(), key=lambda kv: kv[1].start):
        fsl = fidp.orbpair_maps[pair]
        w = csl.stop - csl.start
        if fsl.stop - fsl.start != 8 * w:
            raise ValueError('full spinor features without complex doubling')
        uu.append(np.arange(fsl.start, fsl.start + w))
        dd.append(np.arange(fsl.start + 3 * w, fsl.start + 4 * w))
    return np.concatenate(uu), np.concatenate(dd), fidp.reduced_matrix_element


class FeatureMaps:
    """Feature <-> block index maps of one model mapper; species sets are derived on first use and kept."""

    def __init__(self, cidp, device):
        self.cidp = cidp
        self.device = torch.device(device)
        self.fidp = spinor_mapper(cidp, self.device)
        self.uu, self.dd, self.full_width = layout_maps(cidp, self.fidp)
        self.width = len(self.uu)
        self.norb = {x: int(self.fidp.norbs[x]) for x in self.fidp.norbs}
        self._sets = {}

    def _coded(self, species, blocks, overlap=None):
        """Run block_to_feature on one node per species and one edge per ordered species pair."""
        pairs = [(a, b) for a in range(len(species)) for b in range(len(species))]
        data = {K.ATOMIC_NUMBERS_KEY: torch.tensor([Z_OF[x] for x in species], dtype=torch.long, device=self.device),
                K.EDGE_INDEX_KEY: torch.tensor([[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long,
                                               device=self.device),
                K.EDGE_CELL_SHIFT_KEY: torch.tensor([[q + 1, 0, 0] for q in range(len(pairs))], dtype=torch.float64,
                                                    device=self.device),
                K.POSITIONS_KEY: torch.zeros((len(species), 3), dtype=torch.float64, device=self.device)}
        block_to_feature(data, self.fidp, blocks=blocks, overlap_blocks=overlap, output_dtype=torch.float64)
        return data, pairs

    def _derive(self, species, wh):
        """Encode maps (padded blocks -> compact features) and decode maps (compact delta -> padded spinor blocks)."""
        pairs = [(a, b) for a in range(len(species)) for b in range(len(species))]
        norb = [self.norb[x] for x in species]
        # encode: H entry (r, c) of a [2wh, 2wh] block carries 1 + 2(r 2wh + c) (real) and 2 + 2(r 2wh + c) (imaginary),
        # S entry (r, c) of a [wh, wh] block carries 1 + (r wh + c); a compact feature then names the flat index of
        # view_as_real(H) (resp. S) it reads, -1 = zero.
        hcode = np.arange(4 * wh * wh, dtype=np.float64).reshape(2 * wh, 2 * wh)
        scode = np.arange(wh * wh, dtype=np.float64).reshape(wh, wh)
        hb, sb = {}, {}
        for a, n in enumerate(norb):
            hb['%d_%d_0_0_0' % (a, a)] = (1 + 2 * hcode[:2 * n, :2 * n]) + 1j * (2 + 2 * hcode[:2 * n, :2 * n])
            sb['%d_%d_0_0_0' % (a, a)] = 1 + scode[:n, :n]
        for q, (a, b) in enumerate(pairs):
            hb['%d_%d_%d_0_0' % (a, b, q + 1)] = (1 + 2 * hcode[:2 * norb[a], :2 * norb[b]]) + \
                1j * (2 + 2 * hcode[:2 * norb[a], :2 * norb[b]])
            sb['%d_%d_%d_0_0' % (a, b, q + 1)] = 1 + scode[:norb[a], :norb[b]]
        data, _ = self._coded(species, hb, sb)
        uu = torch.as_tensor(self.uu, device=self.device)

        def enc(f):
            c = f[:, uu]
            if bool((c != torch.round(c)).any()) or bool((c < 0).any()):
                raise ValueError('block to feature conversion is not a pure selection')
            return torch.round(c).to(torch.int64) - 1

        hn, he = enc(data[K.NODE_FEATURES_KEY]), enc(data[K.EDGE_FEATURES_KEY])
        sn, se = enc(data[K.NODE_OVERLAP_KEY]), enc(data[K.EDGE_OVERLAP_KEY])
        # decode: compact column c carries c + 1; written to uu and dd as compact_delta_to_full does
        full = torch.zeros((1, self.full_width), dtype=torch.float64, device=self.device)
        code = torch.arange(1, self.width + 1, dtype=torch.float64, device=self.device)
        full[0, uu] = code
        full[0, torch.as_tensor(self.dd, device=self.device)] = code
        fd = {K.ATOMIC_NUMBERS_KEY: torch.tensor([Z_OF[x] for x in species], dtype=torch.long, device=self.device),
              K.NODE_FEATURES_KEY: full.repeat(len(species), 1), K.EDGE_FEATURES_KEY: full.repeat(len(pairs), 1),
              K.EDGE_INDEX_KEY: torch.tensor([[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long,
                                             device=self.device),
              K.EDGE_CELL_SHIFT_KEY: torch.tensor([[q + 1, 0, 0] for q in range(len(pairs))], dtype=torch.float64,
                                                  device=self.device)}
        blocks = feature_to_block(fd, self.fidp)

        def dec(b):
            b = np.asarray(b)
            if np.any(b.real != np.rint(b.real)) or np.any(b.imag != np.rint(b.imag)):
                raise ValueError('feature to block conversion is not a pure selection')
            re, im = np.rint(b.real).astype(np.int64) - 1, np.rint(b.imag).astype(np.int64) - 1
            out = np.full((2, 2 * wh, 2 * wh), -1, np.int64)
            out[0, :b.shape[0], :b.shape[1]] = re
            out[1, :b.shape[0], :b.shape[1]] = im
            return out

        dn = np.stack([dec(blocks['%d_%d_0_0_0' % (a, a)]) for a in range(len(species))])
        de = np.stack([dec(blocks['%d_%d_%d_0_0' % (a, b, q + 1)]) for q, (a, b) in enumerate(pairs)])
        T = lambda x: torch.as_tensor(x, device=self.device)
        return dict(enc_hn=hn, enc_he=he, enc_sn=sn, enc_se=se, dec_n=T(dn), dec_e=T(de), norb=T(np.asarray(norb)))

    def maps(self, species, wh):
        key = (tuple(species), int(wh))
        if key not in self._sets:
            self._sets[key] = self._derive(list(species), int(wh))
        return self._sets[key]

    def encode(self, species, spec, keys, rows, H, S):
        """Compact uu-real features of the blocks at ``rows`` of padded tables H [m, 2wh, 2wh] and S [m, wh, wh].

        species: sorted symbols; spec [n]: species index of every atom; keys [k, 5]: the block keys (nodes or edges)
        whose features are wanted, rows [k]: their rows in H and S. Returns float32 (H features, S features) [k, width].
        """
        wh = S.shape[-1]
        m = self.maps(species, wh)
        ns = len(species)
        node = keys[:, 0] == keys[:, 1]
        node &= (keys[:, 2:] == 0).all(1)
        hf = torch.view_as_real(H).reshape(len(H), -1)
        sf = S.reshape(len(S), -1)
        out_h = torch.zeros((len(keys), self.width), dtype=torch.float32, device=H.device)
        out_s = torch.zeros((len(keys), self.width), dtype=torch.float32, device=H.device)
        pid = torch.where(node, spec[keys[:, 0]], ns + spec[keys[:, 0]] * ns + spec[keys[:, 1]])
        hmaps = torch.cat([m['enc_hn'], m['enc_he']])
        smaps = torch.cat([m['enc_sn'], m['enc_se']])
        for q in torch.unique(pid).tolist():
            it = torch.nonzero(pid == q, as_tuple=True)[0]
            r = rows[it]
            for f, maps, out in ((hf, hmaps, out_h), (sf, smaps, out_s)):
                idx = maps[q]
                v = f[r[:, None], idx.clamp(min=0)[None, :]]
                out[it] = torch.where(idx[None, :] >= 0, v, torch.zeros_like(v)).to(torch.float32)
        return out_h, out_s

    def decode(self, species, spec, keys, pred, wh):
        """Padded spinor blocks [k, 2wh, 2wh] (complex64) of a compact delta [k, width] at ``keys``."""
        m = self.maps(species, wh)
        ns = len(species)
        node = (keys[:, 0] == keys[:, 1]) & (keys[:, 2:] == 0).all(1)
        pid = torch.where(node, spec[keys[:, 0]], ns + spec[keys[:, 0]] * ns + spec[keys[:, 1]])
        dmaps = torch.cat([m['dec_n'], m['dec_e']])          # [ns + ns^2, 2, 2wh, 2wh]
        z = torch.cat([pred, torch.zeros((len(pred), 1), dtype=pred.dtype, device=pred.device)], 1)
        out = torch.zeros((len(keys), 2 * wh, 2 * wh), dtype=torch.complex64, device=pred.device)
        for q in torch.unique(pid).tolist():
            it = torch.nonzero(pid == q, as_tuple=True)[0]
            re, im = dmaps[q, 0].reshape(-1), dmaps[q, 1].reshape(-1)
            re = torch.where(re < 0, torch.full_like(re, self.width), re)
            im = torch.where(im < 0, torch.full_like(im, self.width), im)
            v = z[it]
            out[it] = torch.complex(v[:, re], v[:, im]).reshape(len(it), 2 * wh, 2 * wh)
        return out
