from __future__ import annotations

import sqlite3
from pathlib import Path

from .core import Recognition


class EventStore:
    def __init__(self, database: str) -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                image_path TEXT
            )"""
        )
        self.connection.commit()

    def add(self, event: Recognition) -> int:
        cursor = self.connection.execute(
            "INSERT INTO recognitions (plate, confidence, timestamp, image_path) VALUES (?, ?, ?, ?)",
            (event.plate, event.confidence, event.timestamp, event.image_path),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def close(self) -> None:
        self.connection.close()

