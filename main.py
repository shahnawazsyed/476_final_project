"""
main.py

Entry point for running the final project.

Responsibilities:
1. Load development or test data (JSON format).
2. Call the agent for each instance.
3. Save results (e.g., data/outputs.json).
4. Ensure efficient execution (<20 API calls per instance).
"""

import json
from agent import run_agent



def main():
    input_path = "data/cse476_final_project_dev_data.json"
    output_path = "outputs/output.json"

    with open(input_path, "r") as f:
        inputs = json.load(f)
    results = []
    for input in inputs:
        prompt = input["input"]
        #print("Trying", prompt)
        response = run_agent(prompt)
        results.append({"prompt": prompt, "response": response})
        #TODO: call agent.py with the prompt
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()