"""Serve the app locally for a quick check:  python tests/serve_local.py  ->  http://localhost:8000"""
import os, sys, importlib.util
from http.server import HTTPServer, BaseHTTPRequestHandler
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bindu", os.path.join(root, "api", "bindu.py"))
bindu = importlib.util.module_from_spec(spec); spec.loader.exec_module(bindu)

class Local(bindu.handler):
    def do_GET(self):
        body = open(os.path.join(root, "index.html"), "rb").read()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}"); HTTPServer(("", port), Local).serve_forever()