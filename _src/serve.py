# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6", "jinja2>=3.1", "pillow>=10", "opencv-python-headless>=4.9,<5"]
# ///
"""Local preview server: edit content.yaml, hit refresh, see the change.

- Rebuilds index.html on page load whenever content.yaml, the template, the build
  script or the photo changed since the last build.
- Tells the browser not to cache anything, so CSS edits show up on a plain refresh too.
- If the build fails (say, a YAML indentation slip), the browser shows the error.

Run with `make site` from the workspace root, or:  uv run jpacilo.github.io/_src/serve.py [port]
"""

import importlib
import sys
import threading
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

sys.dont_write_bytecode = True   # no __pycache__ folder in the site
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

SITE = build.SITE
OUTPUT = build.OUTPUT
lock = threading.Lock()


def sources() -> list[Path]:
    files = [build.CONTENT, SITE / "style.css", *build.TEMPLATE_DIR.glob("*.html"), build.TEMPLATE_DIR / "build.py"]
    try:  # the photo lives wherever content.yaml points
        photo = (yaml.safe_load(build.CONTENT.read_text(encoding="utf-8")) or {}).get("photo")
        if photo:
            files.append(SITE / photo)
    except Exception:
        pass  # a broken YAML file is reported by the build itself
    return [f for f in files if f.exists()]


def stale() -> bool:
    if not OUTPUT.exists():
        return True
    built = OUTPUT.stat().st_mtime
    return any(f.stat().st_mtime > built for f in sources())


def rebuild() -> str | None:
    """Rebuild if needed. Returns an error message, or None when the page is fine."""
    with lock:
        if not stale():
            return None
        try:
            importlib.reload(build)   # pick up edits to build.py without a restart
            build.main()
        except SystemExit as e:
            if e.code not in (0, None):
                return str(e.code)
        except Exception:
            return traceback.format_exc()
    return None


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            error = rebuild()
            if error:
                body = f"Build failed, so the page wasn't updated:\n\n{error}\n".encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; build messages still print


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    rebuild()
    handler = partial(Handler, directory=str(SITE))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        raise SystemExit(f"Port {port} is busy. Stop the other server, or run: make site PORT={port + 1}")
    print(f"Serving http://localhost:{port}  (edits rebuild on refresh; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
