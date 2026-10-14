"""Static prototype server: no application endpoints or brewing code."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1:] == ["--help"]:
            print("description: Serve only prototype files at http://127.0.0.1:8766.\nhelp: Run python serve.py after npm ci --ignore-scripts and python prepare.py.")
            return 0
        print('error: "Unknown arguments"\nhelp: "Run python serve.py or python serve.py --help"')
        return 2
    print('url: "http://127.0.0.1:8766"\nserver_computation: false', flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8766), partial(SimpleHTTPRequestHandler, directory=str(HERE)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
