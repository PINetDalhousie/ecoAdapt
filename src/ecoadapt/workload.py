"""Trace-driven workloads.

Azure traces give arrival times and input/output lengths but no text, so each
request is paired with a ShareGPT (Conversation) or InstructCoder (Coding)
example of matching length, to get realistic speculative acceptance."""
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# ShareGPT conversations [0, PRIOR_SPLIT) fit the acceptance prior only;
# evaluation prompts come from the rest
PRIOR_SPLIT = 5000
IC_CONCAT = (1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34, 48)
IC_PER_LEVEL = 4000


@dataclass
class Request:
    req_id: str
    arrival_s: float
    prompt_ids: list
    max_tokens: int

    @property
    def prompt_len(self):
        return len(self.prompt_ids)


def _seconds(ts):
    # "2024-05-12 00:00:00.001163+00:00" / "2023-11-16 18:15:46.6805900"
    base = datetime.fromisoformat(ts[:19]).timestamp()
    frac = ts[20:].split("+")[0] if ts[19:20] == "." else ""
    return base + (float("0." + frac) if frac else 0.0)


def read_trace(path, n, start_row=0, max_total=None):
    """n consecutive requests of an Azure trace -> (arrival_s, in_len, out_len)."""
    t, li, lo = [], [], []
    with open(path, newline="") as fh:
        for k, row in enumerate(csv.DictReader(fh)):
            if k < start_row:
                continue
            a, b = int(row["ContextTokens"]), int(row["GeneratedTokens"])
            if a < 1 or b < 1 or (max_total and a + b > max_total):
                continue
            t.append(_seconds(row["TIMESTAMP"]))
            li.append(a)
            lo.append(b)
            if len(t) == n:
                break
    t = np.array(t)
    return t - t[0], np.array(li), np.array(lo)


def read_lengths(path):
    """All (input, output) lengths of a trace, e.g. to fit the horizon predictor."""
    li, lo = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = int(row["ContextTokens"]), int(row["GeneratedTokens"])
            if a >= 1 and b >= 1:
                li.append(a)
                lo.append(b)
    return np.array(li), np.array(lo)


def sharegpt_pairs(path, split="eval"):
    """(id, prompt, reference reply); a prompt is the conversation prefix up to a
    human turn."""
    blob = json.loads(Path(path).read_text())
    lo, hi = (0, PRIOR_SPLIT) if split == "prior" else (PRIOR_SPLIT, len(blob))
    for idx in range(lo, min(hi, len(blob))):
        turns = blob[idx].get("conversations", [])
        prefix = []
        for t, turn in enumerate(turns):
            prefix.append(turn.get("value", ""))
            if turn.get("from") not in ("human", "user") or t + 1 >= len(turns):
                continue
            reply = turns[t + 1].get("value", "")
            if len(prefix[-1]) >= 16 and len(reply) >= 16:
                yield f"sg:{idx}:{t}", "\n\n".join(prefix), reply


def instructcoder_pairs(path):
    """Long coding inputs are built from consecutive InstructCoder examples."""
    blob = json.loads(Path(path).read_text())
    n = len(blob)
    for k in IC_CONCAT:
        stride = max(k, (n - k) // IC_PER_LEVEL)
        for start in range(0, n - k, stride):
            exs = blob[start:start + k]
            if any(len(e.get("instruction", "")) < 8 or len(e.get("output", "")) < 8 for e in exs):
                continue
            yield (f"ic:{start}:{k}",
                   "\n\n# ---\n\n".join(f"{e['instruction']}\n\n{e.get('input', '')}" for e in exs),
                   "\n\n# ---\n\n".join(e["output"] for e in exs))


def _pairs(corpus, path):
    return sharegpt_pairs(path) if corpus == "sharegpt" else instructcoder_pairs(path)


def index_corpus(corpus, path, tokenizer, batch=512):
    """[(id, prompt_len, reply_len)] for every candidate example."""
    out, buf = [], []

    def flush():
        if buf:
            p = tokenizer([b[1] for b in buf], add_special_tokens=False)["input_ids"]
            r = tokenizer([b[2] for b in buf], add_special_tokens=False)["input_ids"]
            out.extend((b[0], len(x), len(y)) for b, x, y in zip(buf, p, r) if len(x) >= 8)
            buf.clear()

    for item in _pairs(corpus, path):
        buf.append(item)
        if len(buf) >= batch:
            flush()
    flush()
    return out


def match(in_len, out_len, cands, w_in=1.0, w_out=1.0, reuse_penalty=0.02):
    """Nearest example under d = w_in|p - L_in|/L_in + w_out|r - L_out|/L_out,
    with a small penalty so few examples do not serve most requests."""
    cin = np.array([c[1] for c in cands], dtype=float)
    cout = np.array([c[2] for c in cands], dtype=float)
    uses = np.zeros(len(cands))
    picked, err = [], []
    for li, lo in zip(in_len, out_len):
        ein = np.abs(cin - li) / li
        d = w_in * ein + w_out * np.abs(cout - lo) / lo + reuse_penalty * uses
        j = int(np.argmin(d))
        uses[j] += 1
        picked.append(cands[j][0])
        err.append(ein[j])
    return picked, np.array(err)


def tokenize_selected(corpus, path, tokenizer, wanted):
    want, found = set(wanted), {}
    for eid, prompt, _ in _pairs(corpus, path):
        if eid in want and eid not in found:
            found[eid] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(found) == len(want):
                break
    return found


def save_workload(path, arrival_s, out_len, prompts):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, (a, o, p) in enumerate(zip(arrival_s, out_len, prompts)):
            fh.write(json.dumps(dict(req_id=f"r{i:05d}", arrival_s=round(float(a), 6),
                                     max_tokens=int(o), prompt_ids=list(map(int, p)))) + "\n")


def load_workload(path, rps=None, n=None):
    """Replay requests; scaling all inter-arrival gaps by one factor sets the mean
    rate to `rps` while keeping the burst pattern of the trace window."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = rows[:n] if n else rows
    t = np.array([r["arrival_s"] for r in rows])
    t = t - t[0]
    if rps and t[-1] > 0:
        t = t * (len(t) / t[-1]) / rps
    return [Request(r["req_id"], float(a), r["prompt_ids"], int(r["max_tokens"]))
            for r, a in zip(rows, t)]
