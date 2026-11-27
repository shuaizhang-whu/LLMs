"""Reproduce a stale keep-alive after blocking CPU setup, without model calls."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest

import aiohttp
from scripts.run_successor_full_eval import evaluation_session
from scripts.run_successor_full_eval import reset_trial_cache
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = .04

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.do_GET()


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_reset_waits_for_success_without_forcing_running_requests(self):
        class Response:
            def __init__(self, success):
                self.success = success
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def raise_for_status(self):
                pass
            async def json(self):
                return {"success": self.success}
        class Session:
            def __init__(self, values):
                self.values, self.calls = values, []
            def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return Response(self.values.pop(0))
        session = Session([False, True])
        with patch("scripts.run_successor_full_eval.asyncio.sleep", new=AsyncMock()):
            result = await reset_trial_cache(session, SimpleNamespace(root="http://local"))
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(all(c[1]["json"] == {} for c in session.calls))
        session = Session([False] * 20)
        with patch("scripts.run_successor_full_eval.asyncio.sleep", new=AsyncMock()):
            with self.assertRaisesRegex(RuntimeError, "reset refused"):
                await reset_trial_cache(session, SimpleNamespace(root="http://local"))

    async def test_cpu_preparation_cannot_reuse_an_expired_connection(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        async def request_after_setup(session):
            async with session.get(address, timeout=2) as response:
                await response.read()
            # Simulate synchronous imports/environment loading while the server
            # closes an idle connection; the client cannot process the FIN yet.
            time.sleep(.15)
            async with session.post(address, json={"probe": 1}, timeout=2) as response:
                return await response.json()
        try:
            async with aiohttp.ClientSession(trust_env=False) as pooled:
                with self.assertRaises(aiohttp.ServerDisconnectedError):
                    await request_after_setup(pooled)
            async with evaluation_session() as fresh:
                self.assertEqual(await request_after_setup(fresh), {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
