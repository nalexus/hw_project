"""Request micro-batching for synchronous classification responses."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np


class PredictionBatchQueueFull(RuntimeError):
    """Raised when the API cannot accept more pending prediction requests."""


@dataclass(frozen=True)
class PredictionBatchItem:
    """One queued prediction request and its awaiting response future."""

    text: str
    future: asyncio.Future[str]


class PredictionBatcher:
    """Collect single-document requests into short model inference batches."""

    def __init__(
        self,
        predictor: Any,
        max_delay_ms: int = 100,
        max_batch_size: int = 64,
        max_queue_size: int = 1000,
    ) -> None:
        """Store batching limits and the model predictor."""

        self.predictor = predictor
        self.max_delay_seconds = max_delay_ms / 1000
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue[PredictionBatchItem] = asyncio.Queue(max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Start the background worker once for the current event loop."""

        if self._worker is not None and not self._worker.done():
            return
        self._closed = False
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background worker and fail requests left in the queue."""

        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        self._worker = None
        self._fail_queued(RuntimeError("Prediction batcher stopped"))

    async def predict(self, text: str) -> str:
        """Queue one text and wait for the label assigned by its batch."""

        if self._closed:
            raise RuntimeError("Prediction batcher is stopped")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        try:
            self.queue.put_nowait(PredictionBatchItem(text=text, future=future))
        except asyncio.QueueFull as exc:
            raise PredictionBatchQueueFull("Prediction batch queue is full") from exc
        return await future

    async def _run(self) -> None:
        """Continuously drain request groups and run batched inference."""

        while True:
            first_item = await self.queue.get()
            batch = await self._collect_batch(first_item)
            self._predict_batch(batch)

    async def _collect_batch(
        self, first_item: PredictionBatchItem
    ) -> list[PredictionBatchItem]:
        """Wait briefly for nearby requests so they share one model call."""

        batch = [first_item]
        deadline = asyncio.get_running_loop().time() + self.max_delay_seconds
        while len(batch) < self.max_batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            batch.append(item)
        return batch

    def _predict_batch(self, batch: list[PredictionBatchItem]) -> None:
        """Run one model call and route every label to its original request."""

        try:
            texts = np.array([item.text for item in batch], dtype=object)
            prediction_rows = self.predictor.predict(texts)
            labels = prediction_rows["predicted_label"].tolist()
        except Exception as exc:
            for item in batch:
                self._set_exception(item.future, exc)
            return
        for item, label in zip(batch, labels):
            self._set_result(item.future, str(label))

    def _fail_queued(self, exc: Exception) -> None:
        """Fail queued requests that were never sent to the predictor."""

        while not self.queue.empty():
            item = self.queue.get_nowait()
            self._set_exception(item.future, exc)

    @staticmethod
    def _set_result(future: asyncio.Future[str], label: str) -> None:
        """Set a future result only if the requester is still waiting."""

        if not future.done():
            future.set_result(label)

    @staticmethod
    def _set_exception(future: asyncio.Future[str], exc: Exception) -> None:
        """Set a future exception only if the requester is still waiting."""

        if not future.done():
            future.set_exception(exc)
