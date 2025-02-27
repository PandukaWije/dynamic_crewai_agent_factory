from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, FileReadTool
from crewai.tools import tool
from exa_py import Exa
import os
import re
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

# Create custom Exa search tool
@tool("Exa search and get contents")
def exa_search_tool(query: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    
    exa_api_key = os.environ.get("EXA_API_KEY")
    if not exa_api_key:
        raise ValueError("EXA_API_KEY environment variable not set")
        
    exa = Exa(exa_api_key)
    
    response = exa.search_and_contents(
        query,
        type="neural",
        use_autoprompt=True,
        num_results=5,
        highlights=True
    )
    
    results = []
    for idx, result in enumerate(response.results):
        results.append(f"[SOURCE {idx+1}]\nTitle: {result.title}\nURL: {result.url}\n")
        results.append(f"Highlights:\n{''.join(result.highlights)}\n\n")
    
    return "\n".join(results)

class PromptAnalyzer:
    """Analyzes the user prompt to design an optimal agent team structure."""
    
    def __init__(self, api_key=None, model="gpt-4o"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        system_message = """
        You are an AI team architect who designs optimal agent teams for complex tasks.
        For a given user goal, determine:
        1. The necessary specialized agents (2-5 agents)
        2. Each agent's role, goal, backstory and required tools
        3. The specific tasks each agent should perform
        4. The execution process (sequential or hierarchical)
        
        Available tools: ExaSearchTool (advanced semantic web search), WebsiteSearchTool (website content extraction), 
        FileReadTool (file reading)
        
        IMPORTANT: Do NOT include placeholder variables like {tool_name} in task descriptions.
        Instead write plain text instructions like "Use search tools to gather information".
        
        Return ONLY valid JSON with this structure:
        {
            "agents": [
                {
                    "role": "Role name",
                    "goal": "Agent's goal",
                    "backstory": "Brief backstory",
                    "tools": ["ExaSearchTool", "WebsiteSearchTool", "FileReadTool"]
                }
            ],
            "tasks": [
                {
                    "description": "Task description in plain text without placeholders",
                    "expected_output": "Expected output description",
                    "agent_role": "Which agent performs this"
                }
            ],
            "process": "sequential" or "hierarchical"
        }
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Design an optimal AI agent team for this goal: {prompt}"}
        ]
        
        response = self.llm.invoke(messages)
        content = response.content
        
        # Clean up common formatting issues
        content = content.strip()
        
        # Remove code blocks if present
        if "```json" in content:
            content = content.replace("```json", "", 1)
        if "```" in content:
            content = content.replace("```", "")
        
        # Try to extract JSON using regex if we still have issues
        json_match = re.search(r'({[\s\S]*})', content)
        if json_match:
            content = json_match.group(1)
            
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Content: {content}")
            # Fall back to a minimal valid structure
            return {
                "agents": [
                    {
                        "role": "Researcher",
                        "goal": "Research information related to the task",
                        "backstory": "Expert researcher with analytical skills",
                        "tools": ["ExaSearchTool"]
                    },
                    {
                        "role": "Writer",
                        "goal": "Create content based on research findings",
                        "backstory": "Expert writer skilled at creating engaging content",
                        "tools": []
                    }
                ],
                "tasks": [
                    {
                        "description": f"Research information about: {prompt}",
                        "expected_output": "Comprehensive research findings",
                        "agent_role": "Researcher"
                    },
                    {
                        "description": f"Create content about: {prompt} based on the research",
                        "expected_output": "Engaging and informative content",
                        "agent_role": "Writer"
                    }
                ],
                "process": "sequential"
            }

class AgentFactory:
    """Creates appropriate CrewAI agents based on specifications."""
    
    def __init__(self):
        # Simplified tools map with only the tools we have
        self.tools_map = {
            "ExaSearchTool": exa_search_tool,
            "WebsiteSearchTool": WebsiteSearchTool(),
            "FileReadTool": FileReadTool(),
        }
    
    def create_agents(self, agent_specs: List[Dict]) -> Dict[str, Agent]:
        agents = {}
        for spec in agent_specs:
            # Extract specs
            role = spec["role"]
            goal = spec["goal"]
            backstory = spec["backstory"]
            
            # Get relevant tools
            tools = []
            if "tools" in spec:
                for tool_name in spec["tools"]:
                    if tool_name in self.tools_map:
                        tools.append(self.tools_map[tool_name])
                    else:
                        print(f"Warning: Tool '{tool_name}' not found, skipping")
            
            # Create agent
            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                verbose=True
            )
            
            agents[role] = agent
        
        return agents

class TaskFactory:
    """Creates appropriate CrewAI tasks based on specifications."""
    
    def create_tasks(self, task_specs: List[Dict], agents: Dict[str, Agent]) -> List[Task]:
        tasks = []
        
        for spec in task_specs:
            # Find the agent for this task
            agent_role = spec["agent_role"]
            if agent_role not in agents:
                raise ValueError(f"Agent with role '{agent_role}' not found for task")
            
            # Clean the description to remove any template variables
            description = spec["description"]
            # Remove any {variable} placeholders that might cause template errors
            cleaned_description = re.sub(r'{[^}]*}', '', description)
            
            # Create task
            task = Task(
                description=cleaned_description,
                expected_output=spec["expected_output"],
                agent=agents[agent_role]
            )
            
            tasks.append(task)
        
        return tasks

class CrewFactory:
    """Assembles agents and tasks into a functional CrewAI crew."""
    
    def create_crew(self, agents: Dict[str, Agent], tasks: List[Task], process_type: str) -> Crew:
        # Determine process type
        process = Process.sequential
        if process_type.lower() == "hierarchical":
            process = Process.hierarchical
        
        # Create crew
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=process,
            verbose=True
        )

class DynamicCrewSystem:
    """Complete system for dynamic agent/task creation and execution."""
    
    def __init__(self, openai_api_key=None):
        self.prompt_analyzer = PromptAnalyzer(api_key=openai_api_key)
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()
        self.crew_factory = CrewFactory()
    
    def execute(self, prompt: str, inputs: Dict[str, Any] = None) -> Any:
        """Process user prompt and execute the dynamically created crew."""
        # Get team design from prompt
        team_design = self.prompt_analyzer.analyze(prompt)
        print(f"Generated team design: {json.dumps(team_design, indent=2)}")
        
        # Create agents
        agents = self.agent_factory.create_agents(team_design["agents"])
        
        # Create tasks
        tasks = self.task_factory.create_tasks(team_design["tasks"], agents)
        
        # Create crew
        crew = self.crew_factory.create_crew(
            agents, tasks, team_design.get("process", "sequential")
        )
        
        # Execute crew with any additional inputs
        result = crew.kickoff(inputs=inputs or {})
        
        return result
