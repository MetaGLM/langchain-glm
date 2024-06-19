import asyncio
from typing import AsyncIterable, List, Generic, TypeVar

from tests.assistant.utils import ensure_event_loop

OutputType = TypeVar('OutputType')  # Type variable without bounds

OutputType = TypeVar('OutputType')  # Type variable without bounds


class IteratorCallbackHandler(Generic[OutputType]):
    """Callback handler that returns an iterator."""

    def __init__(self) -> None:
        self.queue = []
        self.done = False

    def put(self, item: OutputType) -> None:
        """Puts an item in the queue."""
        self.queue.append(item)

    def mark_done(self) -> None:
        """Marks the completion of the data input."""
        self.done = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue and self.done:
            raise StopIteration

        loop = asyncio.get_event_loop()
        if not loop.is_running():
            raise RuntimeError("No running event loop in the current thread")

        coroutine = self.queue.pop(0)
        if not asyncio.iscoroutine(coroutine):
            raise TypeError("Queue items must be coroutines")

        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()


async def collect_results(async_gen: AsyncIterable[OutputType],
                          handler: IteratorCallbackHandler[OutputType]) -> List[OutputType]:
    results = []

    async def sample_coroutine(x):
        return x

    async for item in async_gen:
        handler.put(sample_coroutine(item))  # Put each item into the handler's queue.
        results.append(item)
    handler.mark_done()  # Signal that no more items will be put into the queue.
    return results
