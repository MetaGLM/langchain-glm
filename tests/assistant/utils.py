import asyncio


def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Create an event loop to run the async functions synchronously
def run_sync(func, *args, **kwargs):
    loop = ensure_event_loop()
    asyncio.set_event_loop(loop)

    return asyncio.run(func(*args, **kwargs))