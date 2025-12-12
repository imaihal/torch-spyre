# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import functools
import traceback
import sys

@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)

# @torch.library.register_kernel("aten::empty_strided", ["spyre"])
#def spyre__empty_strided(*args, **kwargs):
#   print("IMAIHAL spyre__empty_strided()")
#   print(args)
#   print(kwargs)
#   traceback.print_stack(file=sys.stdout)
#   print("IMAIHAL END")
#   return


# CPU-fallback eager operators


def register_fallback(ops):
    """Register a CPU-fallback kernel for each op."""

    def _decorator(fn):
        for op in ops:
            torch.library.register_kernel(op, ["spyre"])(fn)
        return fn

    return _decorator


def to_cpu():
    """
    Convert any input Tensors on Spyre to CPU

    Note: Apply @to_device() *above* (i.e., before) @to_cpu(), otherwise
    to_device will fail to inter the target device.

        @register_fallback(...)
        @to_device()
        @to_cpu()
        def kernel(...):
            ...
    """

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            def _is_tensor(x):
                return torch.is_tensor(x)

            args = (x.cpu() if _is_tensor(x) else x for x in args)
            kwargs = {k: v.cpu() if _is_tensor(v) else v for k, v in kwargs.items()}
            return fn(*args, **kwargs)

        return _wrapped

    return _decorator


def to_device():
    """
    Convert the result Tensor on CPU to a target Spyre device.

    Target device resolution:
       - If `device` is provided in kwargs: replace it with "cpu.
       - Else infer from tensor inputs; fallback to torch.get_default_device() if none.
    """

    def _ensure_device(args, kwargs):
        if (device := kwargs.get("device", None)) is not None:
            kwargs["device"] = "cpu"
            return device

        tensors = {x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)}
        devices = {t.device for t in tensors}
        if not devices:
            kwargs["device"] = "cpu"
            return torch.get_default_device()

        if len(devices) > 1:
            raise RuntimeError(
                f"Expected all tensors on the same device, but found: {devices}"
            )

        return devices.pop()

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            device = _ensure_device(args, kwargs)
            cpu_result = fn(*args, **kwargs)
            return cpu_result.to(device)

        return _wrapped

    return _decorator


def copy_to_out():
    """
    Handle `out=` semantics:
      - run on CPU
      - copy the result into `out`
      - return `out`
    """

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            out = kwargs.get("out", None)
            if out is None:
                raise RuntimeError("missing required keyword argument: 'out'")
            cpu_result = fn(*args, **kwargs)
            return out.copy_(cpu_result)

        return _wrapped

    return _decorator


#@register_fallback(["aten::_local_scalar_dense"])
# @to_cpu()
# def spyre___local_scalar_dense(*args, **kwargs):
#    print("IMAIHAL call spyre__local_scalar_dense")
#   print(args)
#   print(kwargs)
#   return 

# @register_fallback(
#    ["aten::full.names", "aten::full", "aten::full.out", "aten::full.names_out"]
#)
# @register_fallback(["aten::full"])
# @torch.library.register_kernel("aten::full", ["spyre"])
# @to_device()
def spyre__full(size, fill_value, dtype=None, device=None, pin_memory=False):
    print("IMAIHAL call spyre__full")
    traceback.print_stack(file=sys.stdout)
    print(size)
    print(type(size))
    print(fill_value)
    print(type(fill_value))
    # kwargs.update({"device": "cpu"})
    cpuresult = torch.full(size, fill_value, device="cpu")
    print("cpu result")
    print(cpuresult)
    out = cpuresult.to("spyre")
    print("spyre result")
    # print(out)
    return out 


# @register_fallback(["aten::arange", "aten::arange.start", "aten::arange.start_step"])
# @to_device()
# def spyre__arange(*args, **kwargs):
#     return torch.arange(*args, **kwargs)

# @register_fallback(["aten::arange.out", "aten::arange.start_out"])
#@copy_to_out()
#def spyre__arange_out(*args, out, **kwargs):
#    kwargs.update({"device": "cpu", "dtype": out.dtype, "layout": out.layout})
#    return torch.arange(*args, **kwargs)
