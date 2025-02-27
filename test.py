# test.py
from dynamic_crew import DynamicCrewSystem
import os
import traceback

# Check for API keys
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")
    
if not os.environ.get("EXA_API_KEY"):
    print("Warning: EXA_API_KEY environment variable not set")

# Initialize system
system = DynamicCrewSystem()

def test_prompt(prompt, inputs=None):
    print(f"\n\n{'='*80}")
    print(f"TESTING PROMPT: {prompt}")
    print(f"INPUTS: {inputs or {}}")
    print(f"{'='*80}\n")
    
    try:
        # Execute the system with the prompt
        result = system.execute(prompt, inputs or {})
        
        # Print the result
        print("\nRESULT:")
        print(result)
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return False

# Test with a simple prompt
success = test_prompt(
    "Create a beginner's guide to artificial intelligence with examples and analogies."
)

if success:
    print("\n✅ Test completed successfully!")
else:
    print("\n❌ Test failed.")
