from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from bruteforce_canvas.ui import UIStreamEvent


RUN_ID = "run_001"
REQUEST = b"GET /stream?run_id=run_001 HTTP/1.1\r\nHost: localhost\r\n\r\n"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
PORT_TIMEOUT_SECONDS = 5.0

SERVER_SCRIPT = textwrap.dedent(
    """
    from __future__ import annotations

    import json
    import os
    import select
    import signal
    import sys
    import threading

    from bruteforce_canvas.cli import _make_stream_server
    from bruteforce_canvas.transport import EventBus
    from bruteforce_canvas.ui import UIStreamEvent


    run_id = sys.argv[1]
    event_payloads = json.loads(sys.argv[2])
    bus = EventBus()
    for event_payload in event_payloads:
        bus.publish(UIStreamEvent(**event_payload))

    server = _make_stream_server(run_id, bus, 0)
    stop_requested = False


    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        global stop_requested
        stop_requested = True


    def serve() -> None:
        server.serve_forever(poll_interval=0.1)


    def shutdown() -> None:
        bus.close()
        shutdown_thread = threading.Thread(target=server.shutdown, daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=1.0)
        server.server_close()


    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    print(f"PORT {server.server_address[1]}", flush=True)

    try:
        while not stop_requested:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable and os.read(sys.stdin.fileno(), 1) == b"":
                break
    finally:
        shutdown()
    """
)


def _spawn_server(events: list[UIStreamEvent]) -> tuple[subprocess.Popen[str], int]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_PATH) if not existing_pythonpath else f"{SRC_PATH}{os.pathsep}{existing_pythonpath}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            SERVER_SCRIPT,
            RUN_ID,
            json.dumps([event.model_dump(mode="json") for event in events]),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    port = _read_port(proc)
    time.sleep(0.2)
    return proc, port


def _read_port(proc: subprocess.Popen[str]) -> int:
    if proc.stdout is None:
        raise AssertionError("server subprocess stdout was not captured")

    deadline = time.monotonic() + PORT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            raise AssertionError(f"server subprocess exited before printing PORT: {stderr}")
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([proc.stdout], [], [], min(0.1, remaining))
        if not readable:
            continue
        line = proc.stdout.readline()
        if line.startswith("PORT "):
            return int(line.removeprefix("PORT ").strip())
        if line == "":
            break
    raise AssertionError("server subprocess did not print PORT in time")


def _terminate_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def _connect_stream(port: int) -> socket.socket:
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw_socket.settimeout(2.0)
    raw_socket.connect(("127.0.0.1", port))
    raw_socket.sendall(REQUEST)
    return raw_socket


def _read_until(raw_socket: socket.socket, marker: bytes) -> bytes:
    response = b""
    while marker not in response:
        chunk = raw_socket.recv(4096)
        if not chunk:
            break
        response += chunk
    return response


def _split_headers(response: bytes) -> tuple[str, str]:
    header_bytes, body_bytes = response.split(b"\r\n\r\n", 1)
    return header_bytes.decode("iso-8859-1"), body_bytes.decode("utf-8")


def test_sse_format_is_data_json_double_newline() -> None:
    event = UIStreamEvent(
        event_id="evt-1",
        event_type="run_started",
        run_id=RUN_ID,
        lifecycle_state="running",
        message="started",
        timestamp="2026-01-01T00:00:00Z",
    )
    proc, port = _spawn_server([event])

    try:
        raw_socket = _connect_stream(port)
        try:
            response = _read_until(raw_socket, b"\n\n")
        finally:
            raw_socket.close()

        _, body = _split_headers(response)

        assert body.startswith("data: ")
        json_payload = body[len("data: ") :].strip()
        parsed = json.loads(json_payload)
        assert parsed == event.model_dump(mode="json")
        assert parsed["event_id"] == "evt-1"
        assert body.endswith("\n\n")
    finally:
        _terminate_server(proc)


def test_stream_response_headers() -> None:
    proc, port = _spawn_server([])

    try:
        raw_socket = _connect_stream(port)
        try:
            response = _read_until(raw_socket, b"\r\n\r\n")
        finally:
            raw_socket.close()

        header_text, _ = _split_headers(response)
        header_lines = header_text.split("\r\n")
        status_line = header_lines[0]
        headers = {
            name.lower(): value.strip()
            for line in header_lines[1:]
            if ":" in line
            for name, value in [line.split(":", 1)]
        }

        assert int(status_line.split()[1]) == 200
        assert headers["content-type"] == "text/event-stream"
        assert headers["cache-control"] == "no-cache"
        assert headers["connection"] == "keep-alive"
    finally:
        _terminate_server(proc)


def test_client_disconnect_handled_cleanly() -> None:
    event = UIStreamEvent(
        event_id="seed-1",
        event_type="seed_test",
        run_id=RUN_ID,
        lifecycle_state="running",
        message="seed test event",
        timestamp="2026-01-01T00:00:00Z",
    )
    proc, port = _spawn_server([event])

    try:
        raw_socket = _connect_stream(port)
        try:
            response = _read_until(raw_socket, b"\n\n")
        finally:
            raw_socket.close()

        _, body = _split_headers(response)
        assert body.startswith("data: ")
    finally:
        if proc.poll() is None:
            proc.terminate()

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
        raise
