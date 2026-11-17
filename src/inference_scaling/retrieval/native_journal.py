"""Append-only native request journal for interrupted experimental runs."""
from dataclasses import asdict, is_dataclass
import json
import os
import time

from inference_scaling.retrieval.common import _snapshot


class NativeJournal:
    def __init__(self, backend, tap, path):
        self.backend, self.tap, self.original = backend, tap, backend.sample_batch
        self.stream = path.open('x', encoding='utf-8')
        self.unit, self.sequence, self.io_seconds = 'warmup', 0, 0.
        backend.sample_batch = self.sample_batch

    def write(self, row):
        started = time.perf_counter()
        def convert(value):
            if is_dataclass(value):
                return asdict(value)
            raise TypeError(type(value).__name__)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=convert)+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.io_seconds += time.perf_counter()-started

    def sample_batch(self, requests):
        requests = list(requests)
        index, self.sequence = self.sequence, self.sequence+1
        before = _snapshot(self.backend)
        self.write({'kind':'native_start', 'batch':index, 'unit':self.unit,
                    'requests':[asdict(r) for r in requests], 'snapshot_before':before})
        started = time.perf_counter()
        try:
            samples = self.original(requests)
        except BaseException as error:
            self.write({'kind':'native_failed', 'batch':index, 'unit':self.unit,
                        'error':repr(error), 'snapshot_after':_snapshot(self.backend),
                        'seconds':time.perf_counter()-started,
                        'completed_statistics':{r.request_id:self.tap.rows[r.request_id]
                            for r in requests if r.request_id in self.tap.rows}})
            raise
        # The final measurement contains full hidden diagnostics. This failure
        # journal keeps every native token and SC row without duplicating them.
        self.write({'kind':'native_done', 'batch':index, 'unit':self.unit,
                    'samples':[asdict(s) for s in samples],
                    'statistics':{r.request_id:self.tap.rows[r.request_id][2:]
                        for r in requests if r.request_id in self.tap.rows},
                    'snapshot_after':_snapshot(self.backend), 'seconds':time.perf_counter()-started})
        return samples

    def close(self):
        self.backend.sample_batch = self.original
        self.stream.close()
