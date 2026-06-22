import importlib
import sys


def test_legacy_so2_backend_imports_without_external_so2_cuda_ops():
    sys.modules.pop("so2_cuda_ops", None)

    loader = importlib.import_module("dptb.nn.cuda_ops.extension_loader")
    assert hasattr(loader, "load_cuda_extension")
    assert hasattr(loader, "truthy_env")

    cublas = importlib.import_module("dptb.nn.cublas_grouped_gemm")
    assert hasattr(cublas, "_load_extension")
    assert hasattr(cublas, "grouped_gemm")

    fused = importlib.import_module("dptb.nn.so2_moe_fused_p0")
    assert hasattr(fused, "_load_extension")
    assert hasattr(fused, "try_forward_so2_moe_fused_p0")

    scheduler = importlib.import_module("dptb.nn.so2_cuda_scheduler")
    assert scheduler.SO2CudaSchedulerFunction.__module__ == "dptb.nn.so2_cuda_scheduler"
