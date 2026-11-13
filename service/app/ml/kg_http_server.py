from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from app.ml.kg_paragraph_runner import run_paragraph


class KGRequestHandler(BaseHTTPRequestHandler):
    server_version = "MuslimsMLKG/1.0"

    def _write_json(
        self,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "service": "kg-ml",
                },
            )
            return

        self._write_json(
            404,
            {
                "error": "NotFound",
                "message": "Unknown endpoint",
            },
        )

    def do_POST(self) -> None:
        if self.path != "/extract":
            self._write_json(
                404,
                {
                    "error": "NotFound",
                    "message": "Unknown endpoint",
                },
            )
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )

            if content_length <= 0:
                raise ValueError("request body is required")

            raw_body = self.rfile.read(content_length)

            try:
                payload = json.loads(
                    raw_body.decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "request body must be valid UTF-8 JSON"
                ) from exc

            result = run_paragraph(payload)

            self._write_json(
                200,
                result,
            )

        except ValueError as exc:
            self._write_json(
                400,
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )

        except Exception as exc:
            self._write_json(
                500,
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent HTTP server for Muslims ML KG extraction"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    args = parser.parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        KGRequestHandler,
    )

    print(
        f"KG ML HTTP server listening on "
        f"http://{args.host}:{args.port}",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
