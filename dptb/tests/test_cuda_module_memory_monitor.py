import pytest

torch = pytest.importorskip("torch")

from dptb.plugins.monitor import (
    _is_tensor_product_module,
    _is_torchscript_module,
)


def _module_type(name, module_name):
    return type(
        name,
        (torch.nn.Module,),
        {
            "__module__": module_name,
            "__init__": lambda self: torch.nn.Module.__init__(self),
        },
    )


def test_cuda_module_memory_monitor_selects_non_script_tensor_products():
    E3NNTensorProduct = _module_type(
        "FullyConnectedTensorProduct",
        "e3nn.o3._tensor_product._tensor_product",
    )
    DeePTBWrapper = _module_type(
        "OEQTensorProduct",
        "dptb.nn.embedding.emoles",
    )
    TorchScriptModule = _module_type(
        "RecursiveScriptModule",
        "torch.jit._script",
    )

    assert _is_tensor_product_module(E3NNTensorProduct())
    assert _is_tensor_product_module(DeePTBWrapper())
    assert not _is_tensor_product_module(TorchScriptModule())
    assert _is_torchscript_module(TorchScriptModule())
