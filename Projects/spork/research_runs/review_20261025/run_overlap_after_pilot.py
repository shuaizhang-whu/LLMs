"""Run the bounded hardware preflight after the identified pilot has exited."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
config = json.loads((root / "overlap_dependency.json").read_text())
output = Path(config["preflight"])


def save(value):
    (output / "dependency_state.json").write_text(json.dumps(value, indent=2) + "\n")


save({"state": "waiting_for_live_pilot", **config})
print(f'Waiting for pilot PID {config["pid"]}, start ticks {config["start_ticks"]}', flush=True)
deadline = time.monotonic() + 7200
while True:
    try:
        stat = Path(f'/proc/{config["pid"]}/stat').read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        break
    if stat[19] != config["start_ticks"] or stat[0] == "Z":
        break
    if time.monotonic() >= deadline:
        save({"state": "dependency_still_live_at_deadline", **config})
        raise SystemExit("Pilot is still live; no GPU requests started")
    time.sleep(10)

record = Path(config["pilot"]) / "records/airline_2_r0_spork.json"
seal = record.with_name(record.name + ".seal")
if not record.exists() or not seal.exists() or hashlib.sha256(record.read_bytes()).hexdigest() != json.loads(seal.read_text())["sha256"]:
    save({"state": "dependency_ended_without_valid_sealed_record", **config})
    raise SystemExit("Pilot did not seal its result; inspect it before starting another experiment")
result = json.loads(record.read_text())
save({"state": "running_preflight", "pilot_errors": result["errors"], **config})
print("Pilot exited and sealed its outcome; starting hardware-only preflight", flush=True)
command = ["/mnt/data/zhangshuai/anaconda3/envs/spork/bin/python", "-u",
           "/mnt/data/zhangshuai/LLMs/Projects/spork/scripts/check_actor_overlap.py",
           "--output", config["preflight"], "--launch-delay", "5", "--run"]
status = subprocess.run(command, check=False).returncode
save({"state": "preflight_finished", "exit_code": status, **config})
raise SystemExit(status)
