"""End-to-end check of the live server: N real rounds per language, transcript + timings to stdout."""
import json
import sys
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for lang in ("en", "ar"):
    for i in range(N):
        t0 = time.time()
        r = {"tokens": 0}
        with urllib.request.urlopen(f"http://localhost:8000/api/round?pair={lang}&trained=1", timeout=3600) as resp:
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                e = json.loads(line[6:])
                if e["type"] == "sender_token":
                    r["tokens"] += 1
                elif e["type"] in ("start", "sender_done", "receiver", "round_done", "error", "status"):
                    r.setdefault(e["type"], []).append({k: v for k, v in e.items() if k not in ("code_words", "reference")})
                elif e["type"] == "interceptor":
                    r.setdefault("interceptors", []).append({k: v for k, v in e.items() if k != "detail"})
        r["wall_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(dict(lang=lang, round=i + 1, **r), ensure_ascii=False), flush=True)
