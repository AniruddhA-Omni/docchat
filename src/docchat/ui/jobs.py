"""Background ingestion. Streamlit reruns the script on every interaction, so work runs in a
worker thread owned by a cached resource; the UI polls job progress from a fragment.
Worker threads never call ``st.*``."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from docchat.config import Settings
from docchat.ingest.pipeline import IngestResult, ingest_file
from docchat.services import Services


@dataclass
class Job:
    id: str
    file_name: str
    status: str = "queued"  # queued | running | ready | skipped | error
    message: str = "Waiting"
    result: IngestResult | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None


class IngestJobs:
    def __init__(self, services: Services) -> None:
        self.services = services
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self.jobs: list[Job] = []

    def submit(self, paths: list[Path], settings: Settings, force: bool = False) -> None:
        for path in paths:
            job = Job(id=uuid4().hex, file_name=path.name)
            with self._lock:
                self.jobs.append(job)
            self._pool.submit(self._run, job, path, settings, force)

    def _run(self, job: Job, path: Path, settings: Settings, force: bool) -> None:
        job.status, job.started = "running", time.time()

        def progress(msg: str) -> None:
            job.message = msg

        try:
            job.result = ingest_file(self.services, settings, path, progress, force)
            job.status = job.result.status
            job.message = job.result.error or (
                f"{job.result.n_chunks} chunks, {job.result.n_tables} tables"
                if job.result.status == "ready" else "Already indexed")  # fmt: skip
        except Exception as exc:  # never let a worker die silently
            job.status, job.message = "error", str(exc)[:300]
        job.finished = time.time()

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self.jobs if j.status in ("queued", "running")]

    def recent(self, seconds: float = 120) -> list[Job]:
        now = time.time()
        with self._lock:
            return [j for j in self.jobs if j.finished is None or now - j.finished < seconds]

    def clear_finished(self) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.finished is None]
