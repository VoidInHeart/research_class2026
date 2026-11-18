"""TypeSafe's hosted Choice API. No network activity until choose() is called."""
import json
import os
import urllib.request


class Jev:
    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model, timeout=30, transport=None):
        self.model, self.timeout, self.transport = model, timeout, transport
        self.last_usage = {}
        self.resolved_model = None

    def choose(self, state, options):
        payload = {"model": self.model, "state": state, "questions": {"executor": {
            "type": "choice", "instructions": "Select the executor most likely to generate the correct service, method and arguments.",
            "criteria": options}}}
        self.last_usage = {}
        self.resolved_model = None
        if self.transport:
            answer = self.transport(payload)
        else:
            key = os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise ValueError("Set TYPESAFE_API_KEY before running Jev.")
            req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                answer = json.load(response)
        self.last_usage = answer.get("usage") or {}
        self.resolved_model = answer.get("model")
        item = answer.get("answers", {}).get("executor", {})
        if item.get("type", "choice") != "choice" or item.get("choice") not in options:
            raise ValueError("Jev returned an invalid choice.")
        return item["choice"]


def vector_state(known, weights):
    return json.dumps(dict(zip(known, [round(float(w), 6) for w in weights])), ensure_ascii=False)
