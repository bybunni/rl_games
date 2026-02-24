"""Zero-dependency HTTP server for policy sensitivity analysis."""

import argparse
import json
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from .model_adapter import ModelAdapter

STATIC_DIR = Path(__file__).parent / "static"


class PolicyVizHandler(SimpleHTTPRequestHandler):
    """Routes API requests and serves static files."""

    def __init__(self, *args, adapter, **kwargs):
        self._adapter = adapter
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
            return super().do_GET()
        if self.path == "/api/metadata":
            return self._json_response(self._adapter.get_metadata())
        # Static files
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/infer":
            return self._handle_infer()
        if self.path == "/api/reset_state":
            return self._handle_reset_state()
        self.send_error(404)

    def _handle_infer(self):
        body = self._read_json_body()
        if body is None:
            return
        obs = body.get("obs")
        if obs is None:
            self.send_error(400, "Missing 'obs' field")
            return
        stateful = body.get("stateful", False)
        result = self._adapter.infer(obs, stateful=stateful)
        self._json_response(result)

    def _handle_reset_state(self):
        self._adapter.reset_rnn_state()
        self._json_response({"ok": True})

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self.send_error(400, "Empty body")
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return None

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging: only show errors and API calls
        msg = str(args[0]) if args else ""
        if "/api/" in msg or "error" in msg.lower():
            super().log_message(format, *args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Policy Sensitivity Analysis — Interactive Web Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m tools.policy_viz --model model.onnx "
            "--env-config tools/policy_viz/envs/lunarlander_v2.json"
        ),
    )
    parser.add_argument(
        "--model", required=True, help="Path to ONNX model file"
    )
    parser.add_argument(
        "--env-config",
        default=None,
        help="Path to env config JSON (for feature/action names and ranges)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="HTTP server port (default: 8080)"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    env_config_path = None
    if args.env_config:
        env_config_path = Path(args.env_config)
        if not env_config_path.is_file():
            print(f"Error: env config not found: {env_config_path}", file=sys.stderr)
            sys.exit(1)

    adapter = ModelAdapter(model_path, env_config_path)

    handler_class = partial(PolicyVizHandler, adapter=adapter)
    server = HTTPServer(("", args.port), handler_class)

    meta = adapter.get_metadata()
    env_label = meta.get("env_name") or "unknown env"
    print(f"Policy Viz: {env_label} ({meta['action_type']}, obs_dim={meta['obs_dim']})")
    print(f"Recurrent: {meta['is_recurrent']}")
    print(f"Serving on http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
