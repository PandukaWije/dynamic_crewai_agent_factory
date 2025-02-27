# Dynamic Agent Factory

A flexible framework for dynamically creating AI agent teams based on user prompts. This system leverages CrewAI to automatically design and deploy specialized agent teams that can tackle complex tasks without hardcoding agent roles or tasks.

## 🌟 Features

- **Dynamic Team Construction**: Generates optimal agent teams based on task requirements
- **Automatic Task Assignment**: Creates appropriate tasks and assigns them to specialized agents
- **Intelligent Tool Selection**: Equips agents with the right tools for their specific roles
- **Sophisticated Search**: Integrates with Exa's semantic search for high-quality information retrieval
- **Flexible Execution**: Supports both sequential and hierarchical process flows

## 📋 Requirements

- Python 3.10+ 
- OpenAI API key
- Exa API key (for semantic search functionality)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/dynamic-agent-factory.git
cd dynamic-agent-factory
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install crewai 'crewai[tools]' exa_py langchain-openai
```
or 

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root (using `.env.sample`) :

```
OPENAI_API_KEY=your_openai_api_key
EXA_API_KEY=your_exa_api_key
```

Or set environment variables directly:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export EXA_API_KEY="your_exa_api_key"
```

## 🚀 Usage

### Basic Usage

Create a prompt that describes the task you want to accomplish:

```python
from dynamic_crew import DynamicCrewSystem

# Initialize the system
system = DynamicCrewSystem()

# Execute with a prompt
result = system.execute(
    "Create a comprehensive guide to machine learning for beginners"
)

print(result)
```

### With Additional Inputs

You can provide additional variables to be used in the execution:

```python
result = system.execute(
    "Develop a marketing strategy for a new product launch",
    inputs={
        "product_name": "EcoWash",
        "target_audience": "Environmentally conscious homeowners",
        "launch_date": "March 2026"
    }
)
```

## 🧪 Testing

Run the included test script to verify your installation:

```bash
python test.py
```

This will execute a sample task and display the results.

## 🧠 How It Works

1. **Prompt Analysis**: The system analyzes your prompt to determine the optimal agent team structure
2. **Team Design**: Based on the analysis, it creates a team of specialized agents with appropriate roles and tools
3. **Task Creation**: It designs specific tasks for each agent to perform
4. **Execution**: The agents work together (sequentially or hierarchically) to complete the overall task
5. **Result Delivery**: The final output is returned as the result

## 📊 Example Use Cases

- **Research Reports**: "Research the impact of artificial intelligence on healthcare"
- **Content Creation**: "Create a social media campaign for a sustainable fashion brand"
- **Educational Material**: "Develop lesson plans teaching blockchain to high school students"
- **Market Analysis**: "Analyze the electric vehicle market in Europe and predict trends"
- **Technical Documentation**: "Create comprehensive API documentation for a payment system"

## 🔧 Customization

### Adding Custom Tools

You can extend the `AgentFactory` class to include additional tools:

```python
def __init__(self):
    self.tools_map = {
        "ExaSearchTool": exa_search_tool,
        "WebsiteSearchTool": WebsiteSearchTool(),
        "FileReadTool": FileReadTool(),
        "YourCustomTool": your_custom_tool_function,
    }
```

### Modifying System Prompts

To change how the system designs agent teams, modify the `system_message` in the `PromptAnalyzer` class.

## ⚠️ Troubleshooting

### Common Issues

- **JSON Parsing Errors**: If you see JSON parsing errors, try updating to the latest version of the code which includes robust error handling
- **Missing Tools**: Make sure all required tools are properly imported and registered in the `tools_map`
- **API Rate Limits**: If you're hitting rate limits, consider implementing delays between requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ 
