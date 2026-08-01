"""
Broadcasts every new scoring decision to connected dashboard clients over a
native FastAPI WebSocket (architecture.md section 7: no STOMP/SockJS — that
was Spring-specific), replacing the frontend's earlier 4-second poll with
real push.
"""

import logging

from fastapi import WebSocket

logger = logging.getLogger("websocket_manager")


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning("Dropping dead WebSocket connection: %s", e)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
