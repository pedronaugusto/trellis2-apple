from typing import *
import platform

BACKEND = 'flash_attn'
DEBUG = False

def __detect_defaults():
    """Auto-detect the best dense-attention backend for the current platform.

    `flash_attn` is a CUDA-only extension, so on Apple Silicon it is not
    merely slow, it is unimportable — leaving it as the default made the
    pipeline raise ModuleNotFoundError on the first transformer block.
    torch's SDPA has an MPS implementation, so it is the correct default
    there. Mirrors the same detection in `trellis2/modules/sparse/config.py`
    (which picks the flex_gemm Metal kernels for the *sparse* path).
    """
    global BACKEND
    if platform.system() == 'Darwin' or not __has_cuda():
        BACKEND = 'sdpa'


def __has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def __from_env():
    import os

    global BACKEND
    global DEBUG

    __detect_defaults()

    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')

    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
