from types import SimpleNamespace
import pytest
from dptb.data.dataset.record_pipeline import RecordSchemaValidator,_host

TOKEN='nextham_uureal_729_orbpair_v1'
SCHEMA='deeptb.soc_uureal_named_slots_rme_training_sample/v1'
DIGEST='dcc39e397efb11b1cfb4d56e64720964791418dbf3c06544c02cee0a6758ecce'

def test_soc_named_basis_requires_exact_mapper(monkeypatch):
    mapper=SimpleNamespace(has_soc=True,nextham_uureal_mask=True,reduced_matrix_element=729)
    dataset=SimpleNamespace(type_mapper=mapper)
    record={'basis_fingerprint':TOKEN,'hamiltonian_schema':SCHEMA}
    monkeypatch.setattr(_host,'mapper_basis_fingerprint',lambda m:DIGEST)
    validator=RecordSchemaValidator()
    assert validator.validate_schema_and_basis(dataset,record)==(DIGEST,DIGEST)
    mapper.nextham_uureal_mask=False
    with pytest.raises(ValueError,match='mismatch'):validator.validate_schema_and_basis(dataset,record)
    mapper.nextham_uureal_mask=True
    monkeypatch.setattr(_host,'mapper_basis_fingerprint',lambda m:'a'*64)
    with pytest.raises(ValueError,match='canonical'):validator.validate_schema_and_basis(dataset,record)

def test_soc_named_basis_rejects_wrong_schema(monkeypatch):
    dataset=SimpleNamespace(type_mapper=SimpleNamespace(has_soc=True,nextham_uureal_mask=True,reduced_matrix_element=729))
    with pytest.raises(ValueError,match='mismatch'):
        RecordSchemaValidator().validate_schema_and_basis(dataset,{'basis_fingerprint':TOKEN,'hamiltonian_schema':'wrong'})
