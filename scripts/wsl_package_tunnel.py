"""Temporary TLS-pass-through for this WSL install, bound to the WSL adapter only.

No global proxy changes, TLS interception, credentials, or arbitrary destinations.
Set HTTPS_PROXY for the single installation command, then stop this owned helper.
"""
from __future__ import annotations

import argparse
import ipaddress
import select
import socket
import socketserver
import threading


ALLOWED = frozenset({"pypi.org", "files.pythonhosted.org", "pypi.nvidia.com",
                     "download.pytorch.org", "download-r2.pytorch.org", "github.com",
                     "release-assets.githubusercontent.com", "objects.githubusercontent.com"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--port", type=int, default=18743)
    parser.add_argument("--lifetime", type=int, default=1800)
    args = parser.parse_args()
    bind, client = ipaddress.ip_address(args.bind), ipaddress.ip_address(args.client)
    if (not bind.is_private or not client.is_private or bind.is_unspecified
            or not 1 <= args.lifetime <= 3600 or not 1024 <= args.port <= 65535):
        raise ValueError("explicit private WSL addresses and bounded lifetime required")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            if self.client_address[0] != str(client):
                return
            connection = self.request
            connection.settimeout(20)
            header = bytearray()
            try:
                while not header.endswith(b"\r\n\r\n") and len(header) <= 8192:
                    value = connection.recv(1)
                    if not value:
                        return
                    header.extend(value)
                if not header.endswith(b"\r\n\r\n"):
                    return
                method, authority, _ = header.split(b"\r\n", 1)[0].decode("ascii").split()
                hostname, port = authority.rsplit(":", 1)
                if method != "CONNECT" or hostname not in ALLOWED or port != "443":
                    connection.sendall(b"HTTP/1.0 403 Forbidden\r\n\r\n")
                    return
                with socket.create_connection((hostname, 443), timeout=20) as upstream:
                    connection.sendall(b"HTTP/1.0 200 Connection established\r\n\r\n")
                    upstream.settimeout(60)
                    connection.settimeout(60)
                    while True:
                        ready, _, _ = select.select([connection, upstream], [], [], 60)
                        if not ready:
                            return
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            destination = upstream if source is connection else connection
                            destination.sendall(data)
            except (OSError, ValueError, UnicodeError):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = False

    with Server((str(bind), args.port), Handler) as server:
        timer = threading.Timer(args.lifetime, server.shutdown)
        timer.daemon = True
        timer.start()
        print(f"Temporary package tunnel {bind}:{args.port}; only client {client}; client TLS verification unchanged", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            timer.cancel()


if __name__ == "__main__":
    main()
