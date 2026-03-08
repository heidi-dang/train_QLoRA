import os
import logging
import requests
import time
import threading
import subprocess
from typing import Optional, Tuple
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

        # Provider selection
        # Preferred: TEACHER_PROVIDER (grok|openai|copilot|synthetic)
        # Backward compatible: infer from TEACHER_MODEL
        env_provider = os.environ.get("TEACHER_PROVIDER", "").strip().lower()
        inferred_provider = (os.environ.get("TEACHER_MODEL", "").split("/")[0].strip().lower())
        self.provider = (provider or env_provider or inferred_provider) or "grok"

        # Model selection
        # Primary: TEACHER_MODEL
        # Secondary/Failback: TEACHER_FAILBACK_MODEL
        self.model = model or os.environ.get("TEACHER_MODEL", "github-copilot/gpt-5.3-codex")
        self.failback_model = os.environ.get("TEACHER_FAILBACK_MODEL", "xai/grok-4-1-fast")

        self.last_call_time = 0
        self.min_interval = 1.0
        self._lock = threading.Lock()

        # Copilot uses GitHub auth and does not require TEACHER_API_KEY
        if not self.api_key and self.provider not in ["local", "ollama", "copilot"]:
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

    def _call_copilot(self, prompt: str, code: str) -> Tuple[str, bool]:
        """Call GitHub Copilot CLI as teacher.

        Requires:
        - gh installed
        - gh authenticated (gh auth login)
        - Copilot CLI available via: gh copilot
        """
        code = self._chunk(code)
        full_prompt = (
            "You are a senior engineer and teacher. Produce a high-quality answer.\n\n"
            f"TASK:\n{prompt}\n\n"
            "CONSTRAINTS:\n"
            "- Be precise and correct.\n"
            "- Provide concrete output, not vague advice.\n"
            "- If writing tests, make them executable and cover edge cases.\n"
            "- Keep formatting clean.\n\n"
            f"CODE:\n{code}\n"
        )

        # Copilot CLI doesn't reliably expose token usage; estimate.
        prompt_tokens_estimate = len(full_prompt) // 4

        for attempt in range(3):
            try:
                self._rate_limit()

                # Use Copilot CLI in non-interactive mode.
                # -s: response only
                # --stream off: avoid streaming for easier capture
                cmd = [
                    "gh", "copilot", "--",
                    "--model", self.model,
                    "--output-format", "text",
                    "--stream", "off",
                    "-s",
                    "-p", full_prompt,
                ]

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT[1])
                if proc.returncode == 0:
                    out = (proc.stdout or "").strip()
                    if out:
                        _log_tokens(prompt_tokens_estimate)
                        if telemetry:
                            telemetry.record_api_call("copilot", self.model, prompt_tokens_estimate, 0, True)
                        return out, True

                err = (proc.stderr or "").strip()
                logging.error(f"Copilot CLI failed (attempt={attempt+1}): {err[:300]}")
            except Exception as e:
                logging.error(f"Copilot CLI exception (attempt={attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if telemetry:
            telemetry.record_api_call("copilot", self.model, 0, 0, False)
        return "", False

    def generate(self, prompt: str, code: str) -> str:
        # Step 1: Attempt Primary
        provider = self.provider
        model = self.model
        
        logging.info(f"Primary teacher attempt: {provider}/{model}")
        result, success = self._attempt_call(provider, model, prompt, code)
        if success:
            return result
            
        # Step 2: Attempt Failback
        if self.failback_model:
            fb_provider = self.failback_model.split("/")[0].strip().lower()
            if fb_provider == "xai": fb_provider = "grok"
            logging.warning(f"Primary failed, attempting failback: {fb_provider}/{self.failback_model}")
            result, success = self._attempt_call(fb_provider, self.failback_model, prompt, code)
            if success:
                return result

        logging.error("All teacher attempts failed.")
        return ""

    def _attempt_call(self, provider: str, model: str, prompt: str, code: str) -> Tuple[str, bool]:
        """Internal helper for specific provider calls."""
        orig_model = self.model
        self.model = model  # Temporarily override model for the call
        try:
            if provider == "copilot" or "github" in provider:
                return self._call_copilot(prompt, code)
            elif "grok" in provider or "xai" in provider:
                return self._call_grok(prompt, code)
            elif "openai" in provider or "gpt" in provider:
                return self._call_openai(prompt, code)
            return "", False
        finally:
            self.model = orig_model


def get_client():
    return TeacherAPIClient()
