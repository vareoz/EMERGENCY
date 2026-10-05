"""Panel local para observar y controlar la automejora de NAVROS en tiempo real.

Uso:
    python -m navros dashboard [--run runs/navros] [--port 8777] [--host 127.0.0.1]

Solo usa la librería estándar (no añade dependencias). Sirve una página que se
refresca sola leyendo el estado de la ejecución, con botones para **arrancar**,
**detener** y **borrar** ("desaparecer") el modelo.

Diseño, con franqueza:

* El panel **posee el proceso** de ``improve``: lo lanza como subproceso propio y
  lo mata por PID. Así "detener" y "desaparecer" son fiables.
* La caja «Ingestar» la dispara **una persona** desde el panel (pega un texto o
  una URL y pulsa el botón). **No** forma parte del bucle autónomo: el modelo no
  hace peticiones por su cuenta; es la persona quien decide qué se descarga, y se
  exige que el entrenamiento esté detenido para no escribir el checkpoint a la vez.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_FETCH_BYTES = 8 * 1024 * 1024  # tope de descarga para la ingesta manual de URL
LOG_TAIL_LINES = 60


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    # Windows: consulta el proceso por su handle.
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


def _terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    for _ in range(50):  # hasta ~5 s de margen para que guarde y salga
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("solo se permiten URLs http/https")
    req = urllib.request.Request(url, headers={"User-Agent": "navros-dashboard"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (URL la pone la persona)
        ctype = resp.headers.get("Content-Type", "")
        if ctype and not any(t in ctype for t in ("text", "json", "xml", "csv", "plain")):
            raise ValueError(f"tipo de contenido no textual: {ctype!r}")
        raw = resp.read(MAX_FETCH_BYTES + 1)
    if len(raw) > MAX_FETCH_BYTES:
        raise ValueError(f"el recurso supera el tope de {MAX_FETCH_BYTES // (1024 * 1024)} MB")
    return raw.decode("utf-8", errors="replace")


class Controller:
    """Posee el ciclo de vida del proceso de automejora de una ejecución."""

    def __init__(self, run_dir: Path, log_path: Path | None = None):
        self.run_dir = Path(run_dir)
        self.log_path = Path(log_path) if log_path else self.run_dir / "train.log"
        self.pid_path = self.run_dir / "train.pid"
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.ingest_status = "sin ingestas"
        self.ingesting = False

    # -- proceso ------------------------------------------------------------
    def _read_pid(self) -> int:
        try:
            return int(self.pid_path.read_text().strip())
        except (OSError, ValueError):
            return 0

    def _write_pid(self, pid: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(pid))

    def _clear_pid(self) -> None:
        try:
            self.pid_path.unlink()
        except OSError:
            pass

    def is_alive(self) -> bool:
        if self.proc and self.proc.poll() is None:
            return True
        pid = self._read_pid()
        return bool(pid) and _pid_alive(pid)

    def _state_age(self) -> float | None:
        """Segundos desde la última escritura de state.json (``None`` si no existe)."""
        try:
            return max(0.0, time.time() - (self.run_dir / "state.json").stat().st_mtime)
        except OSError:
            return None

    def external_active(self, window: float = 120.0) -> bool:
        """Un entrenamiento que el panel no arrancó, detectado porque state.json se
        actualizó hace poco. Evita arrancar un segundo escritor sobre la ejecución."""
        if self.is_alive():
            return False
        age = self._state_age()
        return age is not None and age < window

    def start(self) -> str:
        with self.lock:
            if self.is_alive():
                return "el entrenamiento ya está en marcha"
            if self.external_active():
                return ("parece haber un entrenamiento activo sobre esta ejecución (state.json recién "
                        "escrito). No arranco un segundo para no corromper el checkpoint.")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "a", encoding="utf-8")
            logf.write(f"\n===== panel: arranque {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            logf.flush()
            kwargs: dict = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True  # sobrevive a cerrar el panel
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "navros", "improve", "--rounds", "0", "--run", str(self.run_dir)],
                stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), **kwargs,
            )
            self._write_pid(self.proc.pid)
            return f"entrenamiento arrancado (PID {self.proc.pid})"

    def stop(self) -> str:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            pid = self._read_pid()
            if pid and _pid_alive(pid):
                _terminate(pid)
            self.proc = None
            self._clear_pid()
            return "entrenamiento detenido"

    def delete(self) -> str:
        """Detiene y borra por completo la ejecución: desaparece el modelo."""
        self.stop()
        import shutil

        with self.lock:
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir, ignore_errors=True)
            self.ingest_status = "sin ingestas"
            return f"ejecución borrada: {self.run_dir}"

    # -- ingesta manual (la dispara una persona, no el modelo) --------------
    def ingest(self, source: str, is_url: bool) -> str:
        if self.is_alive():
            return "error: detén el entrenamiento antes de ingestar (evita escribir el checkpoint a la vez)"
        if self.ingesting:
            return "ya hay una ingesta en curso"
        if not (source or "").strip():
            return "error: fuente vacía"
        self.ingesting = True
        self.ingest_status = "preparando…"
        threading.Thread(target=self._ingest_worker, args=(source, is_url), daemon=True).start()
        return "ingesta lanzada"

    def _ingest_worker(self, source: str, is_url: bool) -> None:
        try:
            if is_url:
                self.ingest_status = f"descargando {source[:80]}…"
                text = _fetch_url(source.strip())
                origin = source.strip()
            else:
                text, origin = source, "texto pegado"
            tmp = self.run_dir.parent / f"_ingesta_{int(time.time())}.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            self.ingest_status = f"ingestando {len(text):,} caracteres de {origin[:80]}…"
            out = subprocess.run(
                [sys.executable, "-m", "navros", "ingest", str(tmp), "--run", str(self.run_dir)],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800,
            )
            tmp.unlink(missing_ok=True)
            tail = (out.stdout or out.stderr or "").strip().replace("\n", " ")[-300:]
            self.ingest_status = (f"ok ({len(text):,} car.): {tail}" if out.returncode == 0
                                  else f"falló (código {out.returncode}): {tail}")
        except Exception as e:  # noqa: BLE001 — se muestra el motivo en el panel
            self.ingest_status = f"error: {e}"
        finally:
            self.ingesting = False

    # -- lectura de estado --------------------------------------------------
    def _log_tail(self) -> str:
        try:
            data = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(data.splitlines()[-LOG_TAIL_LINES:])

    def snapshot(self) -> dict:
        run = self.run_dir
        state: dict = {}
        try:
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        pool_size = 0
        pool = run / "pool.jsonl"
        if pool.exists():
            try:
                with open(pool, "rb") as f:
                    pool_size = sum(1 for _ in f)
            except OSError:
                pool_size = 0
        history = state.get("history", [])
        latest = history[-1] if history else {}
        alive = self.is_alive()
        return {
            "alive": alive,
            "external": (not alive) and self.external_active(),
            "run_exists": (run / "state.json").exists(),
            "run_dir": str(run),
            "version": state.get("version"),
            "level": state.get("level"),
            "lr": state.get("lr"),
            "k": state.get("k"),
            "plateau": state.get("plateau"),
            "growths": state.get("growths"),
            "created": state.get("created"),
            "params": latest.get("params"),
            "mode": latest.get("mode"),
            "pool_size": pool_size,
            "latest_oracle": latest.get("oracle", {}),
            "history": [
                {"version": h.get("version"), "level": h.get("level"),
                 "score_after": h.get("score_after"), "accepted": h.get("accepted"),
                 "params": h.get("params"), "grew": h.get("grew"),
                 "frontier_yield": h.get("frontier_yield"), "seconds": h.get("seconds")}
                for h in history
            ],
            "log": self._log_tail(),
            "ingest_status": self.ingest_status,
            "ingesting": self.ingesting,
            "now": time.time(),
        }


def _make_handler(ctrl: Controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silencia el log de acceso en consola
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._json(ctrl.snapshot())
            else:
                self._send(404, b"no encontrado", "text/plain; charset=utf-8")

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                body = {}
            if self.path == "/api/start":
                self._json({"msg": ctrl.start()})
            elif self.path == "/api/stop":
                self._json({"msg": ctrl.stop()})
            elif self.path == "/api/delete":
                self._json({"msg": ctrl.delete()})
            elif self.path == "/api/ingest":
                self._json({"msg": ctrl.ingest(body.get("source", ""), bool(body.get("is_url")))})
            else:
                self._json({"error": "ruta desconocida"}, 404)

    return Handler


def serve(run_dir: str | Path = "runs/navros", host: str = "127.0.0.1",
          port: int = 8777, log_path: str | Path | None = None) -> None:
    ctrl = Controller(Path(run_dir), log_path)
    httpd = ThreadingHTTPServer((host, port), _make_handler(ctrl))
    url = f"http://{host}:{port}"
    print(f"[NAVROS] panel en {url}  ·  ejecución: {ctrl.run_dir}")
    print("[NAVROS] Ctrl-C para cerrar el panel (no detiene el entrenamiento).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[NAVROS] panel cerrado.")
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAVROS · panel</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#131824; --panel2:#0f1420; --line:#232c3d;
    --fg:#e6ebf2; --muted:#8a97ad; --accent:#5b9dff; --accent2:#7c5cff;
    --ok:#3fcf8e; --warn:#f5c451; --bad:#ff6b6b; --radius:14px;
  }
  :root:not([data-theme="dark"]){
    @media (prefers-color-scheme: light){
      --bg:#f4f6fb; --panel:#ffffff; --panel2:#f0f3f9; --line:#e2e7f0;
      --fg:#172033; --muted:#5a6678; --accent:#2f6bff; --accent2:#6a44ff;
    }
  }
  :root[data-theme="light"]{
    --bg:#f4f6fb; --panel:#ffffff; --panel2:#f0f3f9; --line:#e2e7f0;
    --fg:#172033; --muted:#5a6678; --accent:#2f6bff; --accent2:#6a44ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    padding:16px;max-width:1120px;margin:0 auto}
  header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:18px}
  h1{font-size:20px;margin:0;letter-spacing:.5px}
  .pill{font-size:12px;font-weight:700;padding:5px 12px;border-radius:999px;text-transform:uppercase;letter-spacing:.6px}
  .pill.on{background:rgba(63,207,142,.16);color:var(--ok);box-shadow:0 0 0 1px rgba(63,207,142,.35) inset}
  .pill.off{background:rgba(255,107,107,.14);color:var(--bad);box-shadow:0 0 0 1px rgba(255,107,107,.3) inset}
  .spacer{flex:1}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px}
  .card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .card .v{font-size:26px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
  .card .v small{font-size:14px;color:var(--muted);font-weight:500}
  .section{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;margin-bottom:16px}
  .section h2{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin:0 0 12px}
  .bars{display:flex;flex-direction:column;gap:6px}
  .bar{display:grid;grid-template-columns:52px 1fr 48px;align-items:center;gap:10px}
  .bar .lab{font-size:12px;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
  .bar .track{background:var(--panel2);border-radius:6px;height:16px;overflow:hidden}
  .bar .fill{height:100%;border-radius:6px;transition:width .4s ease}
  .bar .pct{font-size:12px;font-variant-numeric:tabular-nums;text-align:right}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:10px 18px;cursor:pointer;color:#fff;transition:filter .15s,transform .05s}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .b-start{background:linear-gradient(180deg,var(--ok),#2fae76)}
  .b-stop{background:linear-gradient(180deg,var(--warn),#d8a63a);color:#1c1402}
  .b-del{background:linear-gradient(180deg,var(--bad),#d94c4c)}
  .b-ghost{background:var(--panel2);color:var(--fg);border:1px solid var(--line)}
  pre.log{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px;
    font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-height:320px;overflow:auto;white-space:pre-wrap;margin:0}
  .ing{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .ing input[type=text]{flex:1;min-width:220px;background:var(--panel2);border:1px solid var(--line);
    color:var(--fg);border-radius:10px;padding:10px 12px;font:inherit}
  .ing label{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:6px}
  .note{font-size:12.5px;color:var(--muted);margin-top:10px;line-height:1.5}
  .msg{font-size:13px;color:var(--accent);min-height:18px;margin-top:8px;font-variant-numeric:tabular-nums}
  svg{display:block;width:100%;height:160px}
  .legend{font-size:12px;color:var(--muted);margin-top:6px}
  a.top{color:var(--muted);text-decoration:none;font-size:13px}
</style>
</head>
<body>
<header>
  <h1>NAVROS</h1>
  <span id="status" class="pill off">—</span>
  <span id="sub" class="legend"></span>
  <span class="spacer"></span>
  <a class="top" href="#" id="theme">◐ tema</a>
</header>

<div class="grid" id="tiles"></div>

<div class="section">
  <h2>Controles</h2>
  <div class="controls">
    <button class="b-start" id="start">▶ Arrancar</button>
    <button class="b-stop"  id="stop">⏸ Detener</button>
    <button class="b-del"   id="del">🗑 Desaparecer</button>
    <span class="legend" id="ctrlmsg"></span>
  </div>
  <div class="note">«Desaparecer» detiene el entrenamiento y borra por completo la ejecución
  (pesos, tokenizador, estado y datos auto-generados). No se puede deshacer.</div>
</div>

<div class="section">
  <h2>Exactitud real por nivel (último oráculo)</h2>
  <div class="bars" id="bars"></div>
  <div class="legend">Verde ≥90% · ámbar ≥50% · rojo &lt;50%. La frontera es el nivel más alto en el que entrena.</div>
</div>

<div class="section">
  <h2>Frontera y parámetros por versión</h2>
  <svg id="chart" viewBox="0 0 800 160" preserveAspectRatio="none"></svg>
  <div class="legend"><span style="color:var(--accent)">●</span> frontera (dígitos) &nbsp;
    <span style="color:var(--accent2)">●</span> parámetros</div>
</div>

<div class="section">
  <h2>Ingesta manual <span class="legend">— la disparas tú, no el modelo</span></h2>
  <div class="ing">
    <input type="text" id="src" placeholder="pega texto, o una URL http(s) con licencia clara">
    <label><input type="checkbox" id="isurl"> es URL</label>
    <button class="b-ghost" id="ingest">Ingestar</button>
  </div>
  <div class="msg" id="ingmsg"></div>
  <div class="note">Detén el entrenamiento antes de ingestar. Solo texto; tope 8&nbsp;MB.
  El bucle de automejora solo verifica sumas, así que ingestar prosa no lo hace «más listo» en general
  — para conocimiento general hacen falta corpus grandes en GPU.</div>
</div>

<div class="section">
  <h2>Registro en vivo</h2>
  <pre class="log" id="log">…</pre>
</div>

<script>
const $=s=>document.querySelector(s);
const fmt=n=>(n==null?"—":Number(n).toLocaleString('es-ES'));
const pct=x=>(x==null?"—":Math.round(x*100)+"%");
let followLog=true;

// tema
const root=document.documentElement;
$("#theme").onclick=e=>{e.preventDefault();
  const cur=root.getAttribute('data-theme');
  const next=cur==='light'?'dark':(cur==='dark'?'light':'light');
  root.setAttribute('data-theme',next);
  try{localStorage.setItem('navros-theme',next)}catch(_){}
};
try{const t=localStorage.getItem('navros-theme'); if(t)root.setAttribute('data-theme',t);}catch(_){}

async function post(path,obj){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(obj||{})});
  return r.json();
}
$("#start").onclick=async()=>{$("#ctrlmsg").textContent=(await post('/api/start')).msg; refresh();};
$("#stop").onclick =async()=>{$("#ctrlmsg").textContent=(await post('/api/stop')).msg; refresh();};
$("#del").onclick  =async()=>{
  if(!confirm('¿Borrar por completo la ejecución? El modelo desaparece y no se puede deshacer.'))return;
  $("#ctrlmsg").textContent=(await post('/api/delete')).msg; refresh();
};
$("#ingest").onclick=async()=>{
  const r=await post('/api/ingest',{source:$("#src").value,is_url:$("#isurl").checked});
  $("#ingmsg").textContent=r.msg;
};
$("#log").addEventListener('scroll',()=>{
  const el=$("#log"); followLog=(el.scrollTop+el.clientHeight>=el.scrollHeight-20);
});

function tile(k,v,sub){return `<div class="card"><div class="k">${k}</div>
  <div class="v">${v}${sub?` <small>${sub}</small>`:''}</div></div>`;}

function drawChart(hist){
  const svg=$("#chart"); const W=800,H=160,pad=8;
  if(!hist.length){svg.innerHTML='';return;}
  const xs=hist.map((h,i)=>i);
  const lv=hist.map(h=>h.level||0), pr=hist.map(h=>h.params||0);
  const maxLv=Math.max(...lv,1), maxPr=Math.max(...pr,1);
  const X=i=>pad+(W-2*pad)*(hist.length<2?0.5:i/(hist.length-1));
  const Yl=v=>H-pad-(H-2*pad)*(v/maxLv);
  const Yp=v=>H-pad-(H-2*pad)*(v/maxPr);
  const path=(arr,Y)=>arr.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)).join(' ');
  svg.innerHTML=
    `<path d="${path(pr,Yp)}" fill="none" stroke="var(--accent2)" stroke-width="2" opacity=".85"/>`+
    `<path d="${path(lv,Yl)}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>`;
}

function drawBars(oracle){
  const box=$("#bars");
  const keys=Object.keys(oracle||{}).map(Number).sort((a,b)=>a-b);
  if(!keys.length){box.innerHTML='<div class="legend">sin datos aún</div>';return;}
  box.innerHTML=keys.map(k=>{
    const v=oracle[k]||0, c=v>=0.9?'var(--ok)':(v>=0.5?'var(--warn)':'var(--bad)');
    return `<div class="bar"><div class="lab">${k} díg</div>
      <div class="track"><div class="fill" style="width:${(v*100).toFixed(1)}%;background:${c}"></div></div>
      <div class="pct">${pct(v)}</div></div>`;
  }).join('');
}

function since(ts){ if(!ts)return'—'; let s=Math.max(0,Date.now()/1000-ts);
  const d=Math.floor(s/86400); s-=d*86400; const h=Math.floor(s/3600); s-=h*3600;
  const m=Math.floor(s/60);
  return (d?d+'d ':'')+(h||d?h+'h ':'')+m+'m';}

async function refresh(){
  let s; try{ s=await (await fetch('/api/state')).json(); }catch(_){ return; }
  const on=s.alive||s.external;
  $("#status").className='pill '+(on?'on':'off');
  $("#status").textContent=on?(s.external?'entrenando (externo)':'entrenando'):(s.run_exists?'detenido':'sin modelo');
  $("#sub").textContent=s.run_exists?`ejecución: ${s.run_dir} · activo ${since(s.created)}`:'';
  $("#start").disabled=on; $("#stop").disabled=!s.alive; $("#del").disabled=!s.run_exists;
  $("#tiles").innerHTML=
    tile('Versión', s.version??'—')+
    tile('Frontera', s.level??'—','díg')+
    tile('Parámetros', fmt(s.params))+
    tile('Crecimientos', s.growths??'—')+
    tile('Datos propios', fmt(s.pool_size))+
    tile('Muestras k', s.k??'—')+
    tile('Tasa lr', s.lr!=null?Number(s.lr).toExponential(2):'—')+
    tile('Modo', s.mode||'—');
  drawBars(s.latest_oracle);
  drawChart(s.history||[]);
  const log=$("#log"); log.textContent=s.log||'(sin registro todavía — arranca el entrenamiento)';
  if(followLog) log.scrollTop=log.scrollHeight;
  if(s.ingest_status) $("#ingmsg").textContent=s.ingest_status;
}
refresh(); setInterval(refresh,1500);
</script>
</body>
</html>
"""
