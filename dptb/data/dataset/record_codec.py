"""Lossless record envelopes shared by LMDB preflight and training readers.

Payloads remain trusted local pickle records; compression changes no fields or
row ordering. Zstandard may use its Python package or the system libzstd.
"""

import ctypes
import ctypes.util
import io
import pickle
import zlib


class _NumpyTwoPickleCompat(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def _zstd_system_decompress(payload):
    library = ctypes.util.find_library("zstd")
    if library is None:
        raise ImportError("ZST1 records require zstandard or system libzstd")
    lib = ctypes.CDLL(library)
    lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    lib.ZSTD_decompress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    source = ctypes.create_string_buffer(payload)
    size = lib.ZSTD_getFrameContentSize(source, len(payload))
    if size >= (1 << 64) - 2:
        raise ValueError(
            "ZST1 frame is invalid or lacks a content size; install zstandard for streaming frames"
        )
    target = ctypes.create_string_buffer(size)
    result = lib.ZSTD_decompress(target, size, source, len(payload))
    if lib.ZSTD_isError(result) or result != size:
        raise ValueError("Invalid or truncated ZST1 record")
    return target.raw[:result]


def decompress_record(serialized):
    blob = bytes(serialized)
    if blob.startswith(b"ZL1\0"):
        return zlib.decompress(blob[4:])
    if blob.startswith(b"ZST1"):
        try:
            import zstandard
        except ImportError:
            return _zstd_system_decompress(blob[4:])
        # A context per call avoids sharing mutable state between reader threads.
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob[4:])) as reader:
            return reader.read()
    return blob


def loads_record(serialized):
    serialized = decompress_record(serialized)
    try:
        return pickle.loads(serialized)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        return _NumpyTwoPickleCompat(io.BytesIO(serialized)).load()
