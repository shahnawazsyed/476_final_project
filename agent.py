"""
agent.py

Defines the main agent logic:
- Selects which reasoning strategy to apply for each input example.
- Manages prompts, responses, and postprocessing.
- Calls `call_model_chat_completions()` from api.py indirectly via strategies.
"""

from strategies import chain_of_thought, self_consistency

def run_agent(prompt: str, domain: str) -> str:
    #TODO: add decisioning for which reasoning strategy to use
    if domain == "math":
        result = self_consistency(prompt, True)
    elif domain == "common_sense":
        result = self_consistency(prompt, False)
    else:
        result = chain_of_thought(prompt, False)
    return result