from dynamic_crew import DynamicCrewSystem
import os

# Check for API keys
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")
    
if not os.environ.get("EXA_API_KEY"):
    print("Warning: EXA_API_KEY environment variable not set")

# Initialize system
system = DynamicCrewSystem()

# Test prompts
test_prompts = [
    {
        "name": "Research Report",
        "prompt": "Research the latest advancements in quantum computing and create a detailed report with the most promising applications.",
        "inputs": {}
    },
    {
        "name": "Industry Newsletter",
        "prompt": "Create a weekly newsletter on the latest developments in renewable energy technology with a focus on solar innovations.",
        "inputs": {"target_audience": "Energy industry professionals", "week": "February 20-27, 2025"}
    },
    {
        "name": "Complex Tech Explanation",
        "prompt": "Create an educational guide explaining large language models to high school students, with simple analogies and practical examples.",
        "inputs": {}
    }
]

# Test a specific prompt
def test_specific_prompt(name, prompt, inputs=None):
    print(f"\n\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"PROMPT: {prompt}")
    print(f"INPUTS: {inputs or {}}")
    print(f"{'='*80}\n")
    
    try:
        # Execute the system with the test prompt
        result = system.execute(prompt, inputs or {})
        
        # Print the result
        print("\nRESULT:")
        print(result)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

# Run all tests
def run_all_tests():
    for test in test_prompts:
        test_specific_prompt(test["name"], test["prompt"], test["inputs"])

# Uncomment the test you want to run
# test_specific_prompt("Custom Research", "Research and analyze the impact of artificial intelligence on healthcare diagnostics over the past 5 years.")
run_all_tests()