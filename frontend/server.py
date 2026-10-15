"""Simple static file server for the frontend UI."""

import http.server
import socketserver
from pathlib import Path

PORT = 5500
FRONTEND_DIR = Path(__file__).resolve().parent


class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), FrontendHandler) as httpd:
        print(f"Frontend running at http://127.0.0.1:{PORT}")
        httpd.serve_forever()
