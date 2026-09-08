from __future__ import annotations

import base64
import json
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

JsonHandler = Callable[[dict[str, Any]], dict[str, Any]]
HealthProvider = Callable[[], Any]
_DIRECT_HTTP_OPENER = build_opener(ProxyHandler({}))

WORKER_NOT_READY = "WORKER_NOT_READY"
WORKER_STARTUP_FAILED = "WORKER_STARTUP_FAILED"
WORKER_RPC_UNAVAILABLE = "WORKER_RPC_UNAVAILABLE"
WORKER_PORT_CONFLICT = "WORKER_PORT_CONFLICT"
WORKER_REQUEST_TIMEOUT = "WORKER_REQUEST_TIMEOUT"
WORKER_INTERNAL_ERROR = "WORKER_INTERNAL_ERROR"
TRAINING_FAILED = "TRAINING_FAILED"


WorkerState = Literal["STARTING", "READY", "FAILED", "STOPPING"]


@dataclass(frozen=True)
class WorkerHealth:
    ok: bool
    state: WorkerState
    edge_id: int
    worker_id: str
    message: str = ""
    error_type: str = ""
    run_id: str = ""
    lease_address: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkerHealth":
        state = str(payload.get("state", "READY" if payload.get("ok", True) else "FAILED"))
        if state not in {"STARTING", "READY", "FAILED", "STOPPING"}:
            state = "FAILED"
        return cls(
            ok=bool(payload.get("ok", state == "READY")),
            state=state,  # type: ignore[arg-type]
            edge_id=int(payload.get("edge_id", 0) or 0),
            worker_id=str(payload.get("worker_id", "")),
            message=str(payload.get("message", "") or ""),
            error_type=str(payload.get("error_type", "") or ""),
            run_id=str(payload.get("run_id", "") or ""),
            lease_address=str(payload.get("lease_address", "") or ""),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class JsonRpcError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = WORKER_INTERNAL_ERROR,
        status: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(str(message))
        self.message = str(message)
        self.error_type = str(error_type or WORKER_INTERNAL_ERROR)
        self.status = int(status or 0)
        self.payload = dict(payload or {})


def encode_bytes(value: bytes | bytearray | None) -> str:
    return base64.b64encode(bytes(value or b"")).decode("ascii")


def decode_bytes(value: object) -> bytes:
    if value in (None, ""):
        return b""
    return base64.b64decode(str(value).encode("ascii"))


def post_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    url = f"http://{endpoint}{path}"
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with open_direct(request, timeout=float(timeout)) as response:
            response_data = response.read()
    except HTTPError as exc:
        raw_message = exc.read().decode("utf-8", errors="replace")
        payload = _decode_json_payload(raw_message)
        message = str(payload.get("message", raw_message or str(exc)))
        error_type = str(payload.get("error_type", "")) or _error_type_for_status(exc.code)
        raise JsonRpcError(
            message,
            error_type=error_type,
            status=int(exc.code),
            payload=payload,
        ) from exc
    except (TimeoutError, URLError) as exc:
        raise JsonRpcError(
            str(exc),
            error_type=_classify_transport_error(exc),
        ) from exc
    if not response_data:
        return {}
    return json.loads(response_data.decode("utf-8"))


def open_direct(request, *, timeout: float):
    """Open internal worker RPC URLs without honoring process proxy env vars."""
    return _DIRECT_HTTP_OPENER.open(request, timeout=float(timeout))


class JsonRpcServer:
    def __init__(
        self,
        *,
        listen_address: str,
        routes: dict[str, JsonHandler],
        health_payload: dict[str, Any] | None = None,
        health_provider: HealthProvider | None = None,
        always_available_routes: set[str] | None = None,
    ) -> None:
        host, port_text = listen_address.rsplit(":", 1)
        self.routes = dict(routes)
        self.httpd = _build_http_server(
            host,
            int(port_text),
            self.routes,
            health_payload=dict(health_payload or {}),
            health_provider=health_provider,
            always_available_routes=set(always_available_routes or {"/shutdown"}),
        )
        actual_host, actual_port = self.httpd.server_address
        self.listen_address = f"{actual_host}:{actual_port}"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.httpd.serve_forever,
            name=f"json-rpc-{self.listen_address}",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _build_http_server(
    host: str,
    port: int,
    routes: dict[str, JsonHandler],
    *,
    health_payload: dict[str, Any],
    health_provider: HealthProvider | None,
    always_available_routes: set[str],
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(self._health_payload())
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:
            handler = routes.get(self.path)
            if handler is None:
                self.send_error(404, "not found")
                return
            if health_provider is not None and self.path not in always_available_routes:
                health = WorkerHealth.from_payload(self._health_payload())
                if health.state != "READY" or not health.ok:
                    if health.state == "FAILED":
                        self._send_json(
                            _error_payload(
                                WORKER_STARTUP_FAILED,
                                health.message or "edge worker startup failed",
                            ),
                            status=500,
                        )
                        return
                    self._send_json(
                        _error_payload(
                            WORKER_NOT_READY,
                            health.message or "edge worker is still starting",
                        ),
                        status=503,
                    )
                    return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")
                result = handler(dict(payload or {}))
            except JsonRpcError as exc:
                self._send_json(
                    _error_payload(exc.error_type, exc.message, payload=exc.payload),
                    status=exc.status or 500,
                )
                return
            except Exception as exc:
                self._send_json(
                    _error_payload(_route_error_type(self.path), str(exc)),
                    status=500,
                )
                return
            self._send_json(result)

        def _health_payload(self) -> dict[str, Any]:
            if health_provider is None:
                return {"ok": True, **health_payload}
            try:
                payload = health_provider()
                if isinstance(payload, WorkerHealth):
                    return payload.to_payload()
                return dict(payload)
            except Exception as exc:
                return {
                    "ok": False,
                    "state": "FAILED",
                    "edge_id": int(health_payload.get("edge_id", 0) or 0),
                    "worker_id": str(health_payload.get("worker_id", "")),
                    "message": str(exc),
                    "error_type": WORKER_INTERNAL_ERROR,
                }

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, int(port)), Handler)


def _decode_json_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_payload(
    error_type: str,
    message: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"success": False, "error_type": str(error_type), "message": str(message)}
    if payload:
        result.update({k: v for k, v in payload.items() if k not in result})
    return result


def _error_type_for_status(status: int) -> str:
    if int(status) == 503:
        return WORKER_NOT_READY
    if int(status) >= 500:
        return WORKER_INTERNAL_ERROR
    return WORKER_RPC_UNAVAILABLE


def _route_error_type(path: str) -> str:
    return TRAINING_FAILED if str(path) == "/submit_training_job" else WORKER_INTERNAL_ERROR


def _classify_transport_error(exc: BaseException) -> str:
    text = str(exc).lower()
    reason = getattr(exc, "reason", None)
    errno_value = getattr(reason, "errno", None)
    if errno_value == 98 or "address already in use" in text:
        return WORKER_PORT_CONFLICT
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return WORKER_REQUEST_TIMEOUT
    return WORKER_RPC_UNAVAILABLE
