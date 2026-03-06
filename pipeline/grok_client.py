"""Placeholder Grok client wrapper.

Implement a real Grok/xAI client here. For now, this module provides a wrapper
API that calls pipeline.generate_samples.send_to_teacher so the code path is
consistent when replacing with a real implementation.
"""
import os
import importlib


def _get_send_to_teacher():
    # import lazily to avoid circular import
    m = importlib.import_module('pipeline.generate_samples')
    return getattr(m, 'send_to_teacher')


class GrokClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get('GROK_API_KEY')

    def generate(self, prompt: str, code: str) -> str:
        send = _get_send_to_teacher()
        return send(prompt, code)


def get_client():
    return GrokClient(api_key=os.environ.get('GROK_API_KEY') or '')
