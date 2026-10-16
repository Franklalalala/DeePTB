"""Units of both priors (Rydberg or Hartree atomic units inside, eV out).

    RY        13.605693122994 eV, the Rydberg energy (CODATA 2018)
    BOHR      0.529177210903 Angstrom (CODATA 2018): the H0 prior (pair bank, ABACUS)
    BOHR_GF   0.5291772109 Angstrom: the GF prior, the value of its tables and reference implementation (it differs
              from BOHR in the eleventh digit and is kept so that GF geometries match their tables)
"""
RY = 13.605693122994
BOHR = 0.529177210903
BOHR_GF = 0.5291772109
