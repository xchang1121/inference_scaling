"""Read or opt this benchmark process out of Windows execution-speed EcoQoS.

No global power plan, other process, driver setting or GPU clock is changed.
The optional override lasts only for the current process.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes


class _PowerState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def process_power_state(*, pid: int | None = None, disable_ecoqos: bool = False) -> dict:
    if sys.platform != "win32":
        return {"available": False, "platform": sys.platform}
    if pid is not None and disable_ecoqos:
        raise ValueError("only the current benchmark may change its power policy")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    for name in ("GetProcessInformation", "SetProcessInformation"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        function.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    def snapshot():
        state = _PowerState(1, 0, 0)
        if not kernel.GetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"control_mask": state.ControlMask, "state_mask": state.StateMask}

    try:
        before = snapshot()
        if disable_ecoqos:
            state = _PowerState(1, before["control_mask"] | 1, before["state_mask"] & ~1)
            if not kernel.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state)):
                raise ctypes.WinError(ctypes.get_last_error())
        return {"available": True, "pid": os.getpid() if pid is None else pid,
                "disable_ecoqos": disable_ecoqos, "before": before, "after": snapshot()}
    finally:
        if pid is not None:
            kernel.CloseHandle(handle)
