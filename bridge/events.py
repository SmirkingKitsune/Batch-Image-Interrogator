"""Fan-out of backend events to connected UI clients.

The bridge re-emits the signals MainWindow already wires between tabs
(image_result_ready, database_busy, model loads, streamed transcript text)
as named events. Each connected client gets its own bounded queue; a client
that stops reading loses events rather than growing memory without bound.
"""

import json
import queue
import threading
from typing import Any, List, Optional, Tuple

Event = Tuple[str, str]  # (name, JSON payload)


class EventBus:
    """Thread-safe publish/subscribe for server-sent events."""

    MAX_QUEUED_EVENTS = 20000

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[Optional[Event]]"] = []

    def subscribe(self) -> "queue.Queue[Optional[Event]]":
        subscriber: "queue.Queue[Optional[Event]]" = queue.Queue(maxsize=self.MAX_QUEUED_EVENTS)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[Optional[Event]]") -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def publish(self, name: str, data: Any = None) -> None:
        """Queue an event for every client. Serialization happens once."""
        payload = json.dumps(data, default=_json_default, ensure_ascii=False)
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait((name, payload))
            except queue.Full:
                # A stalled client; dropping keeps the backend responsive.
                pass

    def close(self) -> None:
        """Wake every client so its stream can end."""
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(None)
            except queue.Full:
                pass


def _json_default(value: Any) -> Any:
    """Serialize the few non-JSON types the backend hands around."""
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "value") and value.__class__.__module__.startswith("core"):
        return value.value  # enums such as ProviderPreference
    return str(value)


def to_json(data: Any) -> str:
    return json.dumps(data, default=_json_default, ensure_ascii=False)
