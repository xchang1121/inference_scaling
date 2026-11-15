"""Replay the real inventory function without a device or Docker daemon."""
import ast
import re
import subprocess
from pathlib import Path

import pytest


def inventory_with(inspect):
    source = Path(__file__).resolve().parents[1] / 'scripts/leased_npu_job.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inventory')
    commands = {
        '/usr/local/bin/npu-smi': '| 0 910B3 | OK |\n| 1 910B3 | OK |\n'
            'No running processes found in NPU 0\nNo running processes found in NPU 1',
        'docker': 'ephemeral reserved',
    }
    env = {'command': lambda args: commands[args[0]], 'inspect': inspect,
           're': re, 'subprocess': subprocess}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), env)
    return env['inventory']()


def test_vanished_container_skipped_but_dedicated_reservation_preserved():
    def inspect(name):
        if name == 'ephemeral':
            raise subprocess.CalledProcessError(1, ['docker', 'inspect', name], stderr='Error: No such object: ephemeral')
        return {'HostConfig': {'Devices': [{'PathOnHost': '/dev/davinci1'}]}}
    assert inventory_with(inspect)[0] == [0]


@pytest.mark.parametrize('message', ['permission denied', 'daemon unavailable', ''])
def test_other_inventory_errors_remain_fail_closed(message):
    def inspect(name):
        raise subprocess.CalledProcessError(1, ['docker', 'inspect', name], stderr=message)
    with pytest.raises(subprocess.CalledProcessError):
        inventory_with(inspect)
