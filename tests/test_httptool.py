import json

import pytest

from navros.httptool import (HttpPolicy, HttpRequest, HttpTool, HttpToolError, PolicyError,
                             is_public_ip)

PUBLIC = "93.184.216.34"


def no_dns(host, port):
    raise AssertionError(f"no debería resolverse {host}: la política debía rechazarlo antes")


def local(port, **kw):
    """Política para el servidor de pruebas: http, loopback y su puerto."""
    kw = {"allow_hosts": ("127.0.0.1",), "schemes": ("http",), "ports": (port,),
          "allow_private": True, **kw}
    return HttpPolicy(**kw)


def get(tool, url, **kw):
    return tool.request(HttpRequest("GET", url, **kw))


# -- política de red ----------------------------------------------------------------------------
@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1",
                                "::ffff:10.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::", "224.0.0.1", "no-es-ip"])
def test_non_public_ips(ip):
    assert not is_public_ip(ip)


@pytest.mark.parametrize("ip", [PUBLIC, "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_ips(ip):
    assert is_public_ip(ip)


@pytest.mark.parametrize("ips", [["127.0.0.1"], ["10.0.0.5"], ["169.254.169.254"], ["::1"],
                                 ["::ffff:127.0.0.1"], [PUBLIC, "10.0.0.1"]])
def test_blocks_hosts_resolving_to_private_ips(ips):
    tool = HttpTool(HttpPolicy(allow_hosts=("a.test",)), resolver=lambda h, p: ips)
    with pytest.raises(PolicyError, match="no pública"):
        get(tool, "https://a.test/")


def test_default_policy_denies_everything():
    with pytest.raises(PolicyError, match="host no permitido"):
        get(HttpTool(HttpPolicy(), resolver=no_dns), "https://example.com/")


@pytest.mark.parametrize("url", [
    "http://a.test/",                    # esquema no permitido por defecto
    "ftp://a.test/", "file:///etc/passwd", "gopher://a.test/",
    "https://a.test:8443/",              # puerto no estándar
    "https://user@a.test/", "https://a.test@evil.test/", "https://a.test:pw@evil.test/",
    "https://evil.test/.a.test", "https://a.test.evil.test/", "https://evil.test#.a.test",
    "https://evil.test\\.a.test/",       # barra invertida: confunde a algunos analizadores
    "https://a.test /", "https://a.test/\x00", "https://a.test/\r\nHost: evil.test",
    "https:///a.test", "https://", "https://[::1]/", "https://127.0.0.1/", "https://xn--/",
])
def test_url_tricks_are_rejected_before_any_dns(url):
    tool = HttpTool(HttpPolicy(allow_hosts=("a.test", "*.a.test")), resolver=no_dns)
    with pytest.raises(PolicyError):
        get(tool, url)


def test_wildcard_host_matching():
    p = HttpPolicy(allow_hosts=("*.a.test", "Exacto.B.test."))
    assert p.host_allowed("x.a.test") and p.host_allowed("x.y.a.test") and p.host_allowed("exacto.b.test")
    assert not p.host_allowed("a.test") and not p.host_allowed("xa.test") and not p.host_allowed("a.test.evil")


def test_method_and_header_policy():
    tool = HttpTool(HttpPolicy(allow_hosts=("a.test",)), resolver=no_dns)
    bad = [
        HttpRequest("POST", "https://a.test/", body=b"x"),                              # método
        HttpRequest("get", "https://a.test/"),                                          # minúsculas
        HttpRequest("GET", "https://a.test/", {"authorization": "Bearer robado"}),
        HttpRequest("GET", "https://a.test/", {"Cookie": "s=1"}),
        HttpRequest("GET", "https://a.test/", {"host": "evil.test"}),
        HttpRequest("GET", "https://a.test/", {"content-length": "0"}),
        HttpRequest("GET", "https://a.test/", {"proxy-x": "1"}),
        HttpRequest("GET", "https://a.test/", {"x-a": "1\r\nX-Evil: 1"}),               # inyección de cabecera
        HttpRequest("GET", "https://a.test/", {"x a": "1"}),
        HttpRequest("GET", "https://a.test/", body=b"x"),                               # GET con cuerpo
    ]
    for req in bad:
        with pytest.raises(PolicyError):
            tool.request(req)


def test_request_body_size_limit():
    tool = HttpTool(HttpPolicy(allow_hosts=("a.test",), methods=("POST",), max_request_bytes=10), resolver=no_dns)
    with pytest.raises(PolicyError, match="demasiado grande"):
        tool.request(HttpRequest("POST", "https://a.test/", body=b"x" * 11))


# -- ejecución contra el servidor local ---------------------------------------------------------------
def test_get_roundtrip(server):
    tool = HttpTool(local(server))
    r = get(tool, f"http://127.0.0.1:{server}/echo?a=1&b=dos", headers={"X-Test": "hola"})
    echo = json.loads(r.body)
    assert r.status == 200 and r.text and echo["method"] == "GET" and echo["path"] == "/echo?a=1&b=dos"
    assert echo["headers"]["x-test"] == "hola" and echo["headers"]["user-agent"].startswith("navros")
    assert r.url.endswith("/echo?a=1&b=dos") and r.redirects == []


def test_post_with_body_when_allowed(server):
    tool = HttpTool(local(server, methods=("GET", "POST")))
    r = tool.request(HttpRequest("POST", f"http://127.0.0.1:{server}/echo", {"content-type": "text/plain"}, b"hola"))
    assert json.loads(r.body)["body"] == "hola"


def test_connects_to_the_resolved_ip_not_to_the_name(server):
    """«pinned.test» no existe en el DNS: solo funciona si se conecta a la IP ya resuelta."""
    tool = HttpTool(local(server, allow_hosts=("pinned.test",)), resolver=lambda h, p: ["127.0.0.1"])
    echo = json.loads(get(tool, f"http://pinned.test:{server}/echo").body)
    assert echo["headers"]["host"] == f"pinned.test:{server}"


def test_dns_is_resolved_once_per_hop(server):
    calls = []

    def resolver(host, port):
        calls.append(host)
        return ["127.0.0.1"]
    get(HttpTool(local(server, allow_hosts=("pinned.test",)), resolver=resolver), f"http://pinned.test:{server}/echo")
    assert calls == ["pinned.test"]


def test_redirect_is_followed_and_revalidated(server):
    tool = HttpTool(local(server))
    r = get(tool, f"http://127.0.0.1:{server}/redir")
    assert r.status == 200 and r.url.endswith("/echo?x=1") and len(r.redirects) == 1


def test_redirect_to_a_disallowed_host_is_blocked(server):
    with pytest.raises(PolicyError, match="host no permitido"):
        get(HttpTool(local(server)), f"http://127.0.0.1:{server}/redir-out")


def test_redirect_loop_is_cut(server):
    with pytest.raises(PolicyError, match="demasiadas redirecciones"):
        get(HttpTool(local(server, max_redirects=2)), f"http://127.0.0.1:{server}/loop")


def test_total_deadline_covers_redirect_hops(server):
    tool = HttpTool(local(server, deadline=-1.0))  # ya vencido antes de empezar
    with pytest.raises(HttpToolError, match="tiempo total"):
        get(tool, f"http://127.0.0.1:{server}/redir")


def test_redirects_are_not_followed_for_post(server):
    tool = HttpTool(local(server, methods=("POST",)))
    r = tool.request(HttpRequest("POST", f"http://127.0.0.1:{server}/redir", body=b"x"))
    assert r.status == 302 and r.headers["location"] == "/echo?x=1"


def test_response_size_is_capped(server):
    r = get(HttpTool(local(server, max_response_bytes=100)), f"http://127.0.0.1:{server}/big")
    assert r.truncated and len(r.body) == 100


def test_stops_reading_an_endless_response(server, stream_probe):
    """Con tope de 1 KB no debe descargar los 64 MiB: el tope protege la memoria, no solo el resultado."""
    r = get(HttpTool(local(server, max_response_bytes=1024)), f"http://127.0.0.1:{server}/stream")
    assert r.truncated and len(r.body) == 1024
    assert stream_probe["done"].wait(10)
    assert stream_probe["sent"] < 32 * 1024 * 1024


def test_text_vs_binary(server):
    tool = HttpTool(local(server))
    assert get(tool, f"http://127.0.0.1:{server}/echo").text is not None
    assert get(tool, f"http://127.0.0.1:{server}/bin").text is None


def test_http_errors_are_responses_but_network_errors_are_exceptions(server):
    tool = HttpTool(local(server))
    assert get(tool, f"http://127.0.0.1:{server}/nada").status == 404
    with pytest.raises(HttpToolError, match="red"):
        get(HttpTool(local(1)), "http://127.0.0.1:1/")  # nadie escucha en el puerto 1
    with pytest.raises(HttpToolError, match="resolver"):
        get(HttpTool(local(server, allow_hosts=("x.test",)), resolver=lambda h, p: []), f"http://x.test:{server}/")


def test_credentials_are_never_sent_over_plain_http(server):
    creds = {"127.0.0.1": {"authorization": "Bearer SECRETO"}}
    echo = json.loads(get(HttpTool(local(server, credentials=creds)), f"http://127.0.0.1:{server}/echo").body)
    assert "authorization" not in echo["headers"]


# -- TLS ----------------------------------------------------------------------------------------------
def tls_tool(port, ctx, **kw):
    pol = HttpPolicy(allow_hosts=("secure.test", "other.test", "wrong.test"), ports=(port,), allow_private=True, **kw)
    return HttpTool(pol, resolver=lambda h, p: ["127.0.0.1"], ssl_context=ctx)


def test_https_pinned_connection_and_per_host_credentials(tls_server):
    port, ctx = tls_server
    tool = tls_tool(port, ctx, credentials={"secure.test": {"Authorization": "Bearer SECRETO"}})
    mine = json.loads(get(tool, f"https://secure.test:{port}/echo").body)
    other = json.loads(get(tool, f"https://other.test:{port}/echo").body)
    assert mine["headers"]["authorization"] == "Bearer SECRETO" and mine["headers"]["host"] == f"secure.test:{port}"
    assert "authorization" not in other["headers"]  # la credencial es solo de su host


def test_https_validates_the_certificate_against_the_original_name(tls_server):
    port, ctx = tls_server
    with pytest.raises(HttpToolError, match="certificado"):  # el certificado no cubre wrong.test
        get(tls_tool(port, ctx), f"https://wrong.test:{port}/echo")


def test_https_rejects_untrusted_certificates(tls_server):
    port, _ = tls_server
    tool = tls_tool(port, None)  # contexto por defecto: no confía en el certificado autofirmado
    with pytest.raises(HttpToolError, match="certificado"):
        get(tool, f"https://secure.test:{port}/echo")


# -- auditoría ----------------------------------------------------------------------------------------
def test_audit_log_has_no_query_and_records_blocks(server, tmp_path):
    log = tmp_path / "audit.jsonl"
    tool = HttpTool(local(server), log_path=log)
    get(tool, f"http://127.0.0.1:{server}/echo?token=SECRETO")
    with pytest.raises(PolicyError):
        get(tool, "http://evil.test/")
    raw = log.read_text()
    assert "SECRETO" not in raw
    ok, blocked = (json.loads(line) for line in raw.splitlines())
    assert ok["status"] == 200 and ok["url"].endswith("/echo?…") and blocked["error"].startswith("host no permitido")
