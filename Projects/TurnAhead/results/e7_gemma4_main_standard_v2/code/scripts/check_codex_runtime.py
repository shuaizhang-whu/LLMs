#!/usr/bin/env python3
"""Check local imports and optional server ownership without generating tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--serving-mode", choices=["invariant", "standard"], default="invariant")
    args = parser.parse_args()
    from spork_core import paths
    from spork_core.benchmarks import tau2 as benchmark
    import spork_core.successor_prefetch as successor
    import tau2
    from tau2.utils.utils import DATA_DIR

    locations = {"repo": paths.REPO_ROOT, "tau2_root": paths.TAU2_ROOT,
                 "tau2_data": DATA_DIR, "tau2_import": Path(tau2.__file__),
                 "successor_import": Path(successor.__file__)}
    if any(not Path(p).resolve().is_relative_to(ROOT) for p in locations.values()):
        raise SystemExit(f"External code or data dependency: {locations}")
    for domain in ("airline", "retail"):
        benchmark.load_system_prompt(domain)
        benchmark.load_tools(domain)
        benchmark.get_environment(domain)
    result = {"local_paths": {k: str(v) for k, v in locations.items()},
              "generation_requests": 0}
    if args.server:
        port = int(os.environ.get("SPORK_CHECK_PORT", "8017"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{port}/spork/status", timeout=20) as response:
                status = json.load(response)
        except Exception as error:
            raise SystemExit(f"Codex server is unavailable on port {port}: {error}") from error
        config = status.get("server_config", {})
        if config.get("project_root") != str(ROOT):
            raise SystemExit("Server belongs to another checkout. Stop the previous server and "
                             "launch this project's scripts/serve_qwen32_candidate_checks.sh in tmux vllm.")
        for relative in ("launch_vllm_spork_http.py", "spork_core/vllm029_d3.py"):
            expected = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if config.get("startup_source_sha256", {}).get(relative) != expected:
                raise SystemExit(f"Server loaded different source: {relative}; restart it")
        if (bool(config.get("batch_invariant_requested")) != (args.serving_mode == "invariant")
                or not config.get("prefix_cache_reset_available")
                or not status.get("has_drafter") or status.get("active_requests") != 0
                or status.get("pending_verifications") != 0):
            raise SystemExit("Server configuration or draft-state preflight failed")
        result["server_project_root"] = config["project_root"]
        result["server_port"] = port
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
