import os
import logging
import requests
import time
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pipeline import telemetry
except ImportError:
    telemetry = None

logging.basicConfig(level=logging.INFO)

MAX_CODE_CHUNK = 4000
API_WORKERS = 3
TIMEOUT = (120, 600)
TOKEN_FILE = "/tmp/teacher_tokens.txt"
_token_lock = threading.Lock()


def _log_tokens(tokens):
    if tokens > 0:
        with _token_lock:
            try:
                with open(TOKEN_FILE, "a") as f:
                    f.write(f"{int(time.time())},{tokens}\n")
            except:
                pass


class TeacherAPIClient:
    def __init__(self, api_key=None, provider=None, model=None):
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY")
        self.provider = (provider or os.environ.get("TEACHER_MODEL", "").split("/")[0].lower()) or "grok"
        self.model = model or os.environ.get("TEACHER_MODEL", "grok-4-1-fast")
        self.last_call_time = 0
        self.min_interval = 1.0
        self._lock = threading.Lock()
        if not self.api_key and self.provider not in ["local", "ollama"]:
            logging.warning("No API key, using synthetic")
            self.provider = "synthetic"

    def _rate_limit(self):
        now = time.time()
        with self._lock:
            if now - self.last_call_time < self.min_interval:
                time.sleep(self.min_interval - (now - self.last_call_time))
            self.last_call_time = time.time()

    def _chunk(self, code: str) -> str:
        return code[:MAX_CODE_CHUNK] + "\n... [truncated]" if len(code) > MAX_CODE_CHUNK else code

    def _call_grok(self, prompt: str, code: str) -> tuple:
        code = self._chunk(code)
        logging.info(f"Grok API call len(code)={len(code)}")
        prompt_tokens_estimate = len(prompt) // 4 + len(code) // 4
        for attempt in range(3):
            try:
                self._rate_limit()
                resp = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": [
                        {"role": "system", "content": "You are an expert."},
                        {"role": "user", "content": f"{prompt}\n\n{code}"}],
                        "max_tokens": 1000, "temperature": 0.7},
                    timeout=TIMEOUT)
                if resp.status_code == 200:
                    result = resp.json()
                    usage = result.get("usage", {})
                    actual_prompt = usage.get("prompt_tokens", prompt_tokens_estimate)
                    actual_completion = usage.get("completion_tokens", 0)
                    _log_tokens(actual_prompt + actual_completion)
                    if telemetry:
                        telemetry.record_api_call("grok", self.model, actual_prompt, actual_completion, True)
                    return result["choices"][0]["message"]["content"], True
                else:
                    logging.error(f"Grok error {resp.status_code}")
                    if telemetry:
                        telemetry.record_api_call("grok", self.model, 0, 0, False)
                    return "", False
            except Exception as e:
                logging.error(f"Grok API failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if telemetry:
            telemetry.record_api_call("grok", self.model, 0, 0, False)
        return "", False

    def _call_openai(self, prompt: str, code: str) -> tuple:
        code = self._chunk(code)
        prompt_tokens_estimate = len(prompt) // 4 + len(code) // 4
        for attempt in range(3):
            try:
                self._rate_limit()
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model or "gpt-4", "messages": [
                        {"role": "system", "content": "You are an expert."},
                        {"role": "user", "content": f"{prompt}\n\n{code}"}],
                        "max_tokens": 1000},
                    timeout=TIMEOUT)
                if resp.status_code == 200:
                    result = resp.json()
                    usage = result.get("usage", {})
                    actual_prompt = usage.get("prompt_tokens", prompt_tokens_estimate)
                    actual_completion = usage.get("completion_tokens", 0)
                    _log_tokens(actual_prompt + actual_completion)
                    if telemetry:
                        telemetry.record_api_call("openai", self.model, actual_prompt, actual_completion, True)
                    return result["choices"][0]["message"]["content"], True
            except Exception as e:
                logging.error(f"OpenAI failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if telemetry:
            telemetry.record_api_call("openai", self.model, 0, 0, False)
        return "", False

    def generate(self, prompt: str, code: str) -> str:
        if "grok" in self.provider:
            result, success = self._call_grok(prompt, code)
            if result:
                return result
            result2, _ = self._call_openai(prompt, code)
            return result2
        elif "openai" in self.provider or "gpt" in self.provider:
            result, _ = self._call_openai(prompt, code)
            return result
        logging.error(f"No valid provider: {self.provider}")
        return ""


def get_client():
    return TeacherAPIClient()
