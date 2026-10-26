"""The shutdown diagnostic against a fake server; no SGLang installation or GPU required."""
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import unittest

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics" / "sglang_stop.sh"
FAKE = '''import os, signal, subprocess, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
p = argparse.ArgumentParser()
p.add_argument('--port', type=int)
args, _ = p.parse_known_args()
if os.environ.get('FAKE_FAIL'):
    raise SystemExit(23)
child = subprocess.Popen(['bash', '-c', 'exec -a sglang::detokenizer sleep 120'])
stopping = False
def stop(*_):
    global stopping
    stopping = True
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
def watch():
    child.wait()
    if not stopping:
        os.kill(os.getpid(), signal.SIGTERM)
threading.Thread(target=watch, daemon=True).start()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def do_POST(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"text":"Paris"}')
server = HTTPServer(('127.0.0.1', args.port), Handler)
try:
    server.serve_forever()
finally:
    stopping = True
    child.terminate()
    child.wait(timeout=5)
    server.server_close()
'''


@unittest.skipUnless(shutil.which("setsid") and shutil.which("curl"), "needs Linux setsid and curl")
class ProbeTest(unittest.TestCase):
    def run_probe(self, mode, fail=False):
        with tempfile.TemporaryDirectory(prefix="probe-cpu-") as directory:
            root = Path(directory)
            package = root / "sglang"
            package.mkdir()
            (package / "__init__.py").write_text('__version__ = "fake-cpu"\n')
            (package / "launch_server.py").write_text(FAKE)
            with socket.socket() as port:
                port.bind(("127.0.0.1", 0))
                number = port.getsockname()[1]
            env = {**os.environ, "PYTHONPATH": str(root), "MODE": mode, "MODEL": "fake", "PORT": str(number),
                   "LOG_DIR": str(root / "logs"), "HEALTH_ATTEMPTS": "50", "HEALTH_INTERVAL_S": "0.1",
                   "STOP_WAIT_S": "5"}
            if fail:
                env["FAKE_FAIL"] = "1"
            result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=20)
            return result

    def test_each_variant_finishes_without_killing_the_probe_or_unrelated_processes(self):
        unrelated = subprocess.Popen(["bash", "-c", "exec -a sglang-unrelated sleep 120"], start_new_session=True)
        try:
            for mode in ("clean", "crash", "nodiag", "group"):
                with self.subTest(mode=mode):
                    result = self.run_probe(mode)
                    self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr)
                    self.assertIn("readiness and generation passed", result.stdout)
                    self.assertIn("done, exiting 0", result.stdout)
                    self.assertIsNone(unrelated.poll())
        finally:
            os.killpg(unrelated.pid, signal.SIGTERM)
            unrelated.wait()

    def test_startup_failure_is_not_reported_as_a_successful_stop(self):
        result = self.run_probe("clean", fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("done, exiting 0", result.stdout)
        self.assertIn("server exited before readiness", result.stderr)


def test_probe_job_is_bounded_and_uploads_only_the_script():
    job = yaml.safe_load(SCRIPT.with_suffix(".yaml").read_text())
    assert job["compute"] == {"num_accelerators": 1, "accelerator_type": "GPU_1xH100"}
    assert job["timeout_minutes"] == 30 and job["max_retries"] == 0
    assert job["environment"] == {"unity_catalog_image": "<miles-image>"}
    assert job["env_variables"]["MODE"] == "clean"
    assert job["env_variables"]["MODEL"] == "<staged-model-path>"
    assert job["code_source"] == {"root_path": ".", "include_paths": [SCRIPT.name]}
    assert job["command"] == "bash ${CODE_SOURCE_PATH}/sglang_stop.sh"
