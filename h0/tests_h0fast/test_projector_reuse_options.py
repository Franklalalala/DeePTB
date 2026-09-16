import pytest
from runtime_options import memory_budgets


def test_reuse_budget_default_and_explicit(monkeypatch):
    monkeypatch.delenv('H0_PROJECTOR_REUSE_MAX_MB',raising=False)
    default=memory_budgets()
    assert default['projector_reuse_max_mb']==0
    monkeypatch.setenv('H0_PROJECTOR_REUSE_MAX_MB','256')
    enabled=memory_budgets()
    assert enabled['projector_reuse_max_mb']==256
    assert {k:v for k,v in default.items() if k!='projector_reuse_max_mb'}=={k:v for k,v in enabled.items() if k!='projector_reuse_max_mb'}


@pytest.mark.parametrize('value',['nan','inf','-1','bad'])
def test_invalid_reuse_budget(monkeypatch,value):
    monkeypatch.setenv('H0_PROJECTOR_REUSE_MAX_MB',value)
    with pytest.raises(ValueError):memory_budgets()
