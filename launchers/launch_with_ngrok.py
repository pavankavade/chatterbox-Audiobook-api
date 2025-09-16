"""Launcher: start uvicorn (optional) and open an ngrok tunnel to expose the API.

Usage:
    python launch_with_ngrok.py --start-server    # start uvicorn and ngrok
    python launch_with_ngrok.py                   # just open ngrok to existing server

Note: If you want to use an ngrok authtoken, set NGROK_AUTHTOKEN environment variable or run
`pyngrok.ngrok.set_auth_token(<token>)` before creating the tunnel.
"""
import argparse
import os
import subprocess
import time
from pyngrok import ngrok

LOCAL_PORT = 8000


def start_uvicorn():
    # start uvicorn in a background process
    cmd = [
        "python",
        "-m",
        "uvicorn",
        "api_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(LOCAL_PORT),
    ]
    # Run in the src directory so imports resolve
    cwd = os.path.join(os.path.dirname(__file__), "..", "src")
    print("Starting uvicorn in background (cwd=", cwd, ")")
    proc = subprocess.Popen(cmd, cwd=cwd)
    time.sleep(1)
    return proc


def open_tunnel(port=LOCAL_PORT):
    # Optionally, set auth token from env
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    print(f"Opening ngrok tunnel to http://127.0.0.1:{port} ...")
    http_tunnel = ngrok.connect(addr=port, bind_tls=True)
    print("Public URL:", http_tunnel.public_url)
    return http_tunnel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-server", action="store_true", help="Start local uvicorn server before opening the tunnel")
    args = parser.parse_args()

    proc = None
    try:
        if args.start_server:
            proc = start_uvicorn()

        tunnel = open_tunnel(LOCAL_PORT)
        print("Tunnel established. Press Ctrl-C to exit and close tunnel.")

        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass
        if proc:
            proc.terminate()