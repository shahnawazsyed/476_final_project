"""
agent.py

Defines the main agent logic:
- Selects which reasoning strategy to apply for each input example.
- Manages prompts, responses, and postprocessing.
- Calls `call_model_chat_completions()` from api.py indirectly via strategies.
"""

from api import call_model_chat_completions

def run_agent(input: str) -> str:
    #TODO: add decisioning for which reasoning strategy to use
    return call_model_chat_completions(input)["text"]