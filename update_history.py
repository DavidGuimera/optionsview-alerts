import csv
from pathlib import Path

SNAPSHOT = Path("optionsview_alerts_scan.csv")
HISTORY = Path("data/optionsview_history.csv")

if not SNAPSHOT.exists():
    raise SystemExit("Snapshot CSV not found")

with SNAPSHOT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("Snapshot CSV empty")

HISTORY.parent.mkdir(parents=True, exist_ok=True)
new_fields = list(rows[0].keys())

existing = []
fields = new_fields
if HISTORY.exists():
    with HISTORY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fields = list(dict.fromkeys((reader.fieldnames or []) + new_fields))

# De-duplicate exact scan/ticker rows in case a job is re-run.
seen = {(r.get("scan_time_utc",""), r.get("ticker",""), r.get("signal_id","")) for r in existing}
for r in rows:
    key = (r.get("scan_time_utc",""), r.get("ticker",""), r.get("signal_id",""))
    if key not in seen:
        existing.append(r)
        seen.add(key)

with HISTORY.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in existing:
        w.writerow({k: r.get(k, "") for k in fields})

print(f"History rows: {len(existing)} -> {HISTORY}")
