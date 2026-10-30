"""LLM client wrapper: disk cache, per-worker rate limiting, daily spend guard.

Every response is appended to cache/llm_cache_<model>.jsonl keyed by a hash of
(model, messages, params), so re-running never re-pays for a call. Thread-safe
(workers within a process) and process-safe (one process per model).
"""
import datetime
import hashlib
import json
import os
import ssl
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg):
        load_dotenv(ROOT / ".env")
        self.cfg = cfg
        # One cache + spend file per model, so models can run in parallel processes.
        # All files are read; the daily cap applies to the sum across models.
        self.cache_dir = ROOT / cfg["paths"]["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        for path in sorted(self.cache_dir.glob("llm_cache*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # another process is mid-append to this file
                    self.cache[rec["key"]] = rec
        self.clients = {}
        self.lock = threading.Lock()
        self.local = threading.local()  # per-worker pacing
        self.session = {"calls": 0, "cached": 0, "usd": 0.0}

    def client(self, provider):
        with self.lock:
            return self._client(provider)

    def _client(self, provider):
        if provider not in self.clients:
            p = self.cfg["providers"][provider]
            base_url = os.environ[p["base_url_env"]]
            if "path" in p:  # same Foundry host, different provider route
                u = urlparse(base_url)
                base_url = f"{u.scheme}://{u.netloc}{p['path']}"
            self.clients[provider] = OpenAI(
                base_url=base_url,
                api_key=os.environ[p["api_key_env"]],
                max_retries=0,  # retries handled here so backoff is ours
                # OS trust store instead of certifi: corporate TLS inspection re-signs traffic.
                http_client=openai.DefaultHttpxClient(verify=ssl.create_default_context()),
            )
        return self.clients[provider]

    @staticmethod
    def key(model, messages, params):
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def cost(self, model_key, usage):
        price = self.cfg["models"][model_key]["price_per_m"]
        return (usage["input"] * price["input"] + usage["output"] * price["output"]) / 1e6

    def spent_today(self):
        today, total = str(datetime.date.today()), 0.0
        for p in self.cache_dir.glob("spend*.json"):
            data = self._read_json(p)
            total += data.get(today, 0.0) if data else 0.0
        return total

    @staticmethod
    def _read_json(path, tries=5):
        """Another process may be replacing the file; retry briefly, then skip."""
        for _ in range(tries):
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError):
                time.sleep(0.05)
        return None

    def add_spend(self, model_key, usd):
        """Atomic replace so readers in other processes never see a half-written file."""
        path = self.cache_dir / f"spend_{model_key}.json"
        data = (self._read_json(path) or {}) if path.exists() else {}
        today = str(datetime.date.today())
        data[today] = data.get(today, 0.0) + usd
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a reader has it open
                time.sleep(0.05)
        os.replace(tmp, path)

    def request(self, model_key, messages, max_tokens, response_format=None):
        """(model config, API params, cache key) for a call.

        Per-model config can rename the token limit (gpt-5: max_completion_tokens),
        raise it (reasoning models need headroom) and add params (reasoning_effort)."""
        m = self.cfg["models"][model_key]
        params = {m.get("max_tokens_param", "max_tokens"): max(max_tokens, m.get("min_max_tokens", 0))}
        if "temperature" in m:
            params["temperature"] = m["temperature"]
        if response_format:
            params["response_format"] = response_format
        params.update(m.get("extra_params", {}))
        return m, params, self.key(m["model"], messages, params)

    def is_cached(self, model_key, messages, max_tokens, response_format=None):
        return self.request(model_key, messages, max_tokens, response_format)[2] in self.cache

    def chat(self, model_key, messages, max_tokens, response_format=None):
        """Return {"text", "usage", "cached"}. Raises BudgetExceeded at the daily cap."""
        m, params, k = self.request(model_key, messages, max_tokens, response_format)
        if k in self.cache:
            with self.lock:
                self.session["cached"] += 1
            rec = self.cache[k]
            return {"text": rec["text"], "usage": rec["usage"], "cached": True}

        if self.spent_today() >= self.cfg["budget"]["daily_usd_cap"]:
            raise BudgetExceeded(f"daily cap ${self.cfg['budget']['daily_usd_cap']} reached")

        resp = self._call(m, messages, params)
        text = resp.choices[0].message.content or ""
        usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
        details = getattr(resp.usage, "completion_tokens_details", None)
        if details is not None and getattr(details, "reasoning_tokens", None):
            usage["reasoning"] = details.reasoning_tokens
        # Gemini 3 Pro thinks even at low effort and leaves the thinking out of
        # completion_tokens; it only shows in total_tokens. It is billed as output.
        hidden = (resp.usage.total_tokens or 0) - usage["input"] - usage["output"]
        if hidden > 0:
            usage["hidden"] = hidden
            usage["output"] += hidden
        usd = self.cost(model_key, usage)
        rec = {"key": k, "model": m["model"], "messages": messages, "params": params,
               "text": text, "usage": usage, "usd": usd,
               "ts": datetime.datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            with open(self.cache_dir / f"llm_cache_{model_key}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[k] = rec
            self.add_spend(model_key, usd)
            self.session["calls"] += 1
            self.session["usd"] += usd
        return {"text": text, "usage": usage, "cached": False}

    def _call(self, m, messages, params):
        rl = self.cfg["rate_limit"]
        for attempt in range(rl["max_retries"] + 1):
            last = getattr(self.local, "last_call", 0.0)
            wait = rl["min_seconds_between_calls"] - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self.local.last_call = time.time()
            try:
                return self.client(m["provider"]).chat.completions.create(
                    model=m["model"], messages=messages, **params)
            except openai.RateLimitError as e:
                err, backoff = e, 30 * 2 ** attempt
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                err, backoff = e, 5 * 2 ** attempt
            if attempt == rl["max_retries"]:
                raise err
            print(f"  {type(err).__name__}; retry {attempt + 1} in {backoff}s")
            time.sleep(backoff)
