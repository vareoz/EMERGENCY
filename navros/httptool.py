"""Herramienta HTTP para agentes, segura por omisión.

El modelo nunca abre sockets ni ve secretos: escribe una petición como texto y
este módulo decide, según una ``HttpPolicy`` que fija el *operador*, si se
ejecuta. Nada de lo que el modelo diga puede ampliar la política.

Defensas (todas activas con la política por defecto):

- **Lista de hosts permitidos** (``allow_hosts``). Vacía = no se permite nada.
- **Anti-SSRF**: el DNS se resuelve una sola vez, *todas* las direcciones deben
  ser públicas (ni loopback, ni redes privadas, ni link-local, ni los metadatos
  de la nube 169.254.169.254) y se conecta a esa IP concreta, sin volver a
  resolver (sin *DNS rebinding*). TLS se valida contra el nombre original.
- Solo ``https``, métodos ``GET``/``HEAD`` y puertos 80/443, salvo que el
  operador lo amplíe.
- Las redirecciones se siguen a mano y cada salto pasa la política completa.
- El modelo no puede fijar ``authorization``, ``cookie``, ``host``…: las
  credenciales las inyecta esta herramienta, solo para su host y solo por https.
- Límites de tiempo y de tamaño de la respuesta; auditoría en JSONL sin
  consultas (``?…``), cabeceras ni cuerpos.

Solo usa la biblioteca estándar. No usa proxies de entorno (``HTTPS_PROXY``):
fijar la IP exige conectar directamente.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urljoin, urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}
REDIRECTS = frozenset({301, 302, 303, 307, 308})
BODYLESS_METHODS = frozenset({"GET", "HEAD"})
# El modelo no puede fijarlas: o rompen el enmarcado HTTP o llevan credenciales.
FORBIDDEN_HEADERS = frozenset({
    "host", "content-length", "transfer-encoding", "connection", "upgrade", "te", "trailer",
    "expect", "authorization", "proxy-authorization", "cookie",
})
MAX_HEADERS, MAX_HEADER_VALUE = 20, 1024

_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOSTNAME = re.compile(rf"^{_LABEL}(?:\.{_LABEL})*$")
# Prefijos IPv6 que embeben una IPv4 (NAT64, 6to4, Teredo): podrían apuntar a una red privada.
_EMBEDDED_V4 = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))
_TEXT_SUBTYPES = ("json", "xml", "javascript", "x-www-form-urlencoded", "yaml", "csv")


class HttpToolError(Exception):
    """Fallo al ejecutar la petición. El mensaje es seguro para enseñárselo al modelo."""


class PolicyError(HttpToolError):
    """La política del operador no permite esta petición."""


@dataclass
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None


@dataclass
class HttpResponse:
    status: int
    reason: str
    headers: dict[str, str]  # nombres en minúsculas
    body: bytes
    truncated: bool = False  # el cuerpo se recortó a ``max_response_bytes``
    url: str = ""  # URL final, tras redirecciones
    redirects: list[str] = field(default_factory=list)  # URLs intermedias

    @property
    def text(self) -> str | None:
        """Cuerpo como texto, o ``None`` si es binario."""
        mime, _, params = self.headers.get("content-type", "").partition(";")
        mime = mime.strip().lower()
        main, _, sub = mime.partition("/")
        if mime and main != "text" and not any(sub == s or sub.endswith("+" + s) for s in _TEXT_SUBTYPES):
            return None
        m = re.search(r"charset=([\w.-]+)", params, re.I)
        try:
            return self.body.decode(m.group(1) if m else "utf-8", errors="replace" if mime else "strict")
        except (LookupError, UnicodeDecodeError):
            return self.body.decode("utf-8", errors="replace") if mime else None


@dataclass
class HttpPolicy:
    """Lo que el operador permite. Los valores por defecto son los más restrictivos útiles."""

    allow_hosts: tuple[str, ...] = ()  # "api.ejemplo.com" o "*.ejemplo.com" (no incluye el dominio raíz)
    schemes: tuple[str, ...] = ("https",)
    methods: tuple[str, ...] = ("GET", "HEAD")
    ports: tuple[int, ...] = (80, 443)
    allow_private: bool = False  # True solo para pruebas o redes de confianza
    timeout: float = 10.0  # por operación de socket
    deadline: float = 30.0  # total por llamada, incluidas las redirecciones
    max_redirects: int = 3
    max_response_bytes: int = 1_000_000
    max_request_bytes: int = 16_384
    credentials: dict[str, dict[str, str]] = field(default_factory=dict)  # host exacto → {cabecera: valor}
    user_agent: str = "navros-agent/0.1"

    def host_allowed(self, host: str) -> bool:
        for pat in (p.lower().rstrip(".") for p in self.allow_hosts):
            if host == pat or (pat.startswith("*.") and host.endswith(pat[1:])):
                return True
        return False

    def check_shape(self, method: str, headers: dict[str, str], body: bytes | None) -> dict[str, str]:
        """Valida método, cabeceras y cuerpo; devuelve las cabeceras normalizadas (minúsculas)."""
        if method not in self.methods:
            raise PolicyError(f"método no permitido: {method[:16]}")
        if len(headers) > MAX_HEADERS:
            raise PolicyError("demasiadas cabeceras")
        clean = {}
        for name, value in headers.items():
            name = name.lower()
            if not _TOKEN.match(name):
                raise PolicyError("nombre de cabecera inválido")
            if name in FORBIDDEN_HEADERS or name.startswith("proxy-"):
                raise PolicyError(f"cabecera no permitida: {name}")
            if len(value) > MAX_HEADER_VALUE or any(ord(c) < 32 and c != "\t" or ord(c) == 127 for c in value):
                raise PolicyError(f"valor de cabecera inválido: {name}")
            clean[name] = value
        if body is not None:
            if method in BODYLESS_METHODS:
                raise PolicyError(f"{method} no admite cuerpo")
            if len(body) > self.max_request_bytes:
                raise PolicyError("cuerpo de la petición demasiado grande")
        return clean

    def check_url(self, url: str) -> "_Target":
        """Valida la URL contra la política (sin red). Devuelve el destino ya normalizado."""
        if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
            raise PolicyError("URL con espacios o caracteres de control")
        try:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
            if scheme not in self.schemes:
                raise PolicyError(f"esquema no permitido: {scheme[:10] or '(vacío)'}")
            if "@" in parts.netloc:
                raise PolicyError("URL con credenciales incrustadas")
            host = _normalize_host(parts.hostname)
            if not self.host_allowed(host):
                raise PolicyError(f"host no permitido: {host}")
            port = parts.port or DEFAULT_PORTS[scheme]
        except ValueError:  # puerto fuera de rango, IPv6 mal formada…
            raise PolicyError("URL inválida") from None
        if port not in self.ports:
            raise PolicyError(f"puerto no permitido: {port}")
        target = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
        if parts.query:
            target += "?" + quote(parts.query, safe="=&%+:/?@!$'()*,;-._~")
        return _Target(scheme, host, port, target)


@dataclass(frozen=True)
class _Target:
    scheme: str
    host: str
    port: int
    path: str  # ruta + consulta, ya codificada


def _normalize_host(raw: str | None) -> str:
    if not raw:
        raise PolicyError("URL sin host")
    host = raw.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise PolicyError("host inválido") from None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not _HOSTNAME.match(host):  # frena trucos como «malo.com\.permitido.com»
            raise PolicyError("host inválido") from None
    return host


def is_public_ip(ip: str) -> bool:
    """¿Es una dirección enrutable en Internet? (conservador ante la duda)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return is_public_ip(str(addr.ipv4_mapped))
        if any(addr in net for net in _EMBEDDED_V4):
            return False
    return addr.is_global and not addr.is_multicast


def system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


def _connect_any(ips: list[str], port: int, timeout: float) -> socket.socket:
    err: OSError = OSError("sin direcciones")
    for ip in ips:  # todas ya validadas por la política
        try:
            return socket.create_connection((ip, port), timeout)
        except OSError as e:
            err = e
    raise err


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Conecta a una IP ya validada; ``host`` solo se usa para la cabecera Host."""

    def __init__(self, host: str, port: int, ips: list[str], timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._ips = ips

    def connect(self) -> None:
        self.sock = _connect_any(self._ips, self.port, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Igual, y valida el certificado TLS contra el nombre original (SNI incluido)."""

    def __init__(self, host: str, port: int, ips: list[str], timeout: float, context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._ips = ips

    def connect(self) -> None:
        sock = _connect_any(self._ips, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class HttpTool:
    """Ejecuta ``HttpRequest`` bajo una ``HttpPolicy``.

    ``resolver(host, port) -> [ip]`` y ``ssl_context`` se pueden sustituir (pruebas,
    CA propia). ``log_path`` activa la auditoría en JSONL.
    """

    def __init__(self, policy: HttpPolicy, *, resolver: Callable[[str, int], list[str]] = system_resolver,
                 ssl_context: ssl.SSLContext | None = None, log_path: str | Path | None = None):
        self.policy, self.resolver = policy, resolver
        self.ssl_context = ssl_context or ssl.create_default_context()
        self.log_path = Path(log_path) if log_path else None

    def request(self, req: HttpRequest) -> HttpResponse:
        """Ejecuta la petición o lanza ``PolicyError`` / ``HttpToolError``."""
        t0 = time.monotonic()
        try:
            resp = self._request(req, t0 + self.policy.deadline)
        except HttpToolError as e:
            self._audit(req, t0, error=str(e))
            raise
        self._audit(req, t0, resp=resp)
        return resp

    def _request(self, req: HttpRequest, deadline: float) -> HttpResponse:
        p = self.policy
        headers = p.check_shape(req.method, req.headers, req.body)
        url, trail = req.url, []
        for _ in range(p.max_redirects + 1):
            if time.monotonic() > deadline:
                raise HttpToolError("tiempo total agotado")
            target = p.check_url(url)
            resp = self._send(target, req.method, headers, req.body, self._resolve(target), deadline)
            nxt = resp.headers.get("location")
            # Solo se siguen redirecciones de GET/HEAD: reenviar un cuerpo a otro destino no es seguro.
            if resp.status in REDIRECTS and nxt and req.method in BODYLESS_METHODS:
                trail.append(url)
                url = urljoin(url, nxt)
                continue
            resp.url, resp.redirects = url, trail
            return resp
        raise PolicyError("demasiadas redirecciones")

    def _resolve(self, t: _Target) -> list[str]:
        try:
            ips = self.resolver(t.host, t.port)
        except OSError:
            raise HttpToolError("no se pudo resolver el host") from None
        if not ips:
            raise HttpToolError("no se pudo resolver el host")
        if not self.policy.allow_private and not all(is_public_ip(ip) for ip in ips):
            raise PolicyError("el host resuelve a una dirección no pública")
        return ips

    def _send(self, t: _Target, method: str, headers: dict[str, str], body: bytes | None,
              ips: list[str], deadline: float) -> HttpResponse:
        p = self.policy
        hdrs = {"user-agent": p.user_agent, "accept": "*/*", **headers, "connection": "close"}
        if t.scheme == "https":  # las credenciales nunca viajan en claro ni a otro host
            hdrs.update({k.lower(): v for k, v in p.credentials.get(t.host, {}).items()})
            conn = _PinnedHTTPSConnection(t.host, t.port, ips, p.timeout, self.ssl_context)
        else:
            conn = _PinnedHTTPConnection(t.host, t.port, ips, p.timeout)
        try:
            conn.request(method, t.path, body=body, headers=hdrs)
            r = conn.getresponse()
            data = bytearray()
            while len(data) <= p.max_response_bytes:
                if time.monotonic() > deadline:
                    raise HttpToolError("tiempo total agotado")
                chunk = r.read(min(8192, p.max_response_bytes + 1 - len(data)))
                if not chunk:
                    break
                data += chunk
            truncated = len(data) > p.max_response_bytes
            return HttpResponse(r.status, r.reason, {k.lower(): v for k, v in r.getheaders()},
                                bytes(data[:p.max_response_bytes]), truncated)
        except HttpToolError:
            raise
        except ssl.SSLCertVerificationError:
            raise HttpToolError("certificado TLS no válido para ese host") from None
        except TimeoutError:
            raise HttpToolError("tiempo de espera agotado") from None
        except ssl.SSLError:
            raise HttpToolError("error TLS") from None
        except (OSError, http.client.HTTPException) as e:
            raise HttpToolError(f"error de red ({type(e).__name__})") from None
        finally:
            conn.close()

    def _audit(self, req: HttpRequest, t0: float, resp: HttpResponse | None = None,
               error: str | None = None) -> None:
        if self.log_path is None:
            return
        parts = urlsplit(req.url)
        shown = f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}{parts.path}" + ("?…" if parts.query else "")
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": req.method, "url": shown[:300],
                 "ms": round((time.monotonic() - t0) * 1000)}
        entry.update({"status": resp.status, "bytes": len(resp.body)} if resp else {"error": error})
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
