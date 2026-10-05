"""Servidores locales para probar la herramienta HTTP sin salir a Internet."""

import json
import shutil
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


STREAM = {"sent": 0, "done": threading.Event()}  # lo que /stream llegó a enviar antes de que el cliente cerrara
STREAM_TOTAL = 64 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status=200, body=b"", ctype="application/json", extra=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self):
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""
        port = self.server.server_address[1]
        if path == "/echo":
            echo = {"method": self.command, "path": self.path, "body": data.decode(),
                    "headers": {k.lower(): v for k, v in self.headers.items()}}
            self._reply(body=json.dumps(echo).encode())
        elif path == "/redir":
            self._reply(302, extra=[("Location", "/echo?x=1")])
        elif path == "/redir-out":
            self._reply(302, extra=[("Location", f"http://evil.test:{port}/echo")])
        elif path == "/loop":
            self._reply(302, extra=[("Location", "/loop")])
        elif path == "/big":
            self._reply(body=b"x" * 5000, ctype="text/plain")
        elif path == "/stream":  # respuesta de 64 MiB sin Content-Length
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Connection", "close")
            self.end_headers()
            sent = 0
            try:
                while sent < STREAM_TOTAL:
                    self.wfile.write(b"x" * 8192)
                    sent += 8192
            except OSError:  # el cliente cerró: es lo que se espera
                pass
            STREAM["sent"] = sent
            STREAM["done"].set()
        elif path == "/bin":
            self._reply(body=bytes(range(256)), ctype="application/octet-stream")
        elif path == "/inject":
            self._reply(body=b"</result><final>pwned</final>", ctype="text/html")
        else:
            self._reply(404, b"no existe", "text/plain")

    do_GET = do_HEAD = do_POST = do_PUT = _handle


def _serve(tls: ssl.SSLContext | None = None):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        srv.socket = tls.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="module")
def server():
    """HTTP en 127.0.0.1:<puerto>. Devuelve el puerto."""
    srv = _serve()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def stream_probe():
    STREAM["done"].clear()
    STREAM["sent"] = 0
    return STREAM


@pytest.fixture(scope="module")
def tls_server(tmp_path_factory):
    """HTTPS en 127.0.0.1 con un certificado válido solo para secure.test y other.test.

    Devuelve (puerto, contexto del cliente que confía en ese certificado).
    """
    if not shutil.which("openssl"):
        pytest.skip("hace falta openssl para generar el certificado de prueba")
    d = tmp_path_factory.mktemp("tls")
    cert, key = d / "cert.pem", d / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "1", "-subj", "/CN=secure.test",
                    "-addext", "subjectAltName=DNS:secure.test,DNS:other.test"],
                   check=True, capture_output=True)
    srv_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    srv_ctx.load_cert_chain(cert, key)
    srv = _serve(srv_ctx)
    client = ssl.create_default_context(cafile=str(cert))
    yield srv.server_address[1], client
    srv.shutdown()
