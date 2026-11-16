"""Read-only check of the exact legacy training job after its tool cancellation."""
import json
from pathlib import Path
import psutil

root = Path(__file__).resolve().parent
status = json.loads((root/'runs/queue_status.json').read_text(encoding='utf-8'))
matches = []
for process in psutil.process_iter(['pid','name','cmdline']):
    try:
        command = process.info['cmdline'] or []
        if '--output' not in command:
            continue
        output = command[command.index('--output')+1].replace('\\','/').rstrip('/').lower()
        if output.endswith('/runs/esc_direction_full') and any('run.py' in arg for arg in command):
            matches.append(dict(pid=process.pid, command=command))
    except (psutil.Error, ValueError, IndexError):
        continue
report = dict(parent_pid=status['pid'], parent_exists=psutil.pid_exists(status['pid']), matching_legacy_children=matches)
(root/'artifacts/legacy_stop_confirmation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
