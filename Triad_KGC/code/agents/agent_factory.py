### Factory design for building agent instances with predefined template paths ###
### Relationship: Depends on internal `api` module as the agent brain; external modules should call the `run` method ###
### Dependency: Utilizes utility functions from the `utils` package for parsing and deserialization ###

from Triad_KGC.code.agents.api import get_response
from jinja2 import Template
from abc import ABC, abstractmethod
from typing import Type, Dict, Any

import os

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "agent_templates")

EVOLUTION_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "evolution")
VERIFICATION_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "verification")
SCHEMA_EXPLORATION_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "schema_exploration")


def load_template(template_path: str) -> str:
    """Load a template file from the given path."""
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()


class Agent(ABC):
    """Abstract base class for all Agents. All subclasses must implement `create` and `run`."""

    @classmethod
    @abstractmethod
    def create(cls, agent: str) -> "Agent":
        """Factory method to create an agent instance."""
        pass

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Unified interface for external modules to call the agent."""
        pass


class EvolutionAgent(Agent):
    """Agent for the 'evolution' phase, holds paths to completion-related templates."""

    TriplesCompletionAgent: str = os.path.join(EVOLUTION_TEMPLATE_DIR, "triples_completion_agent.txt")
    BackgroundGenerationAgent: str = os.path.join(EVOLUTION_TEMPLATE_DIR, "background_generation_agent.txt")

    def __init__(self, agent: str):
        self.agent = agent

    @classmethod
    def create(cls, agent: str) -> "EvolutionAgent":
        return cls(agent)

    async def run(self, **kwargs: Any) -> Any:
        print(f"[EvolutionAgent] Running {self.agent} with args: {kwargs}")
        template_path = getattr(self, self.agent, None)
        agent_template = load_template(template_path)
        template = Template(agent_template)
        agent_query = template.render(**kwargs)
        response = await get_response(agent_query)
        return response


class VerificationAgent(Agent):
    """Agent for the 'verification' phase, holds paths to various verification-related templates."""

    ClaimVerificationAgent: str = os.path.join(VERIFICATION_TEMPLATE_DIR, "assertion_verification_agent.txt")
    JudgementVerificationAgent: str = os.path.join(VERIFICATION_TEMPLATE_DIR, "relation_existence_verification_agent.txt")
    SelectionVerificationAgent: str = os.path.join(VERIFICATION_TEMPLATE_DIR, "relation_canonicality_verification_agent.txt")
    InterferenceGenerationAgent: str = os.path.join(VERIFICATION_TEMPLATE_DIR, "relation_canonicality_verification_interference_agent.txt")
    DomainKnowledgeGenerationAgent: str = os.path.join(VERIFICATION_TEMPLATE_DIR, "relation_canonicality_verification_domain_agent.txt")

    def __init__(self, agent: str):
        self.agent = agent

    @classmethod
    def create(cls, agent: str) -> "VerificationAgent":
        return cls(agent)

    async def run(self, **kwargs: Any) -> Any:
        print(f"[VerificationAgent] Running {self.agent} with args: {kwargs}")
        template_path = getattr(self, self.agent, None)
        agent_template = load_template(template_path)
        template = Template(agent_template)
        agent_query = template.render(**kwargs)
        return await get_response(agent_query)


class SchemaExplorationAgent(Agent):
    """Agent for the 'schema exploration' phase, holding paths for schema-related prompts."""

    EntityTypeDefineAgent: str = os.path.join(SCHEMA_EXPLORATION_TEMPLATE_DIR, "entity_type_define_agent.txt")
    RelationTypeDefineAgent: str = os.path.join(SCHEMA_EXPLORATION_TEMPLATE_DIR, "relation_type_define_agent.txt")
    EntityClassifyAgent: str = os.path.join(SCHEMA_EXPLORATION_TEMPLATE_DIR, "entity_classify_agent.txt")
    RelationClassifyAgent: str = os.path.join(SCHEMA_EXPLORATION_TEMPLATE_DIR, "relation_classify_agent.txt")

    def __init__(self, agent: str):
        self.agent = agent

    @classmethod
    def create(cls, agent: str) -> "SchemaExplorationAgent":
        return cls(agent)

    async def run(self, **kwargs: Any) -> Any:
        print(f"[SchemaExplorationAgent] Running {self.agent} with args: {kwargs}")
        template_path = getattr(self, self.agent, None)
        agent_template = load_template(template_path)
        template = Template(agent_template)
        agent_query = template.render(**kwargs)
        return await get_response(agent_query)


class AgentFactory:
    """Factory class for creating different types of agent instances."""

    _registry: Dict[str, Type[Agent]] = {
        "evolution": EvolutionAgent,
        "verification": VerificationAgent,
        "schema_exploration": SchemaExplorationAgent
    }

    @staticmethod
    def create_agent(agent_type: str, agent: str) -> Agent:
        """
        Create an agent instance of the specified type.

        :param agent_type: Type of the agent: 'evolution', 'verification', or 'schema_exploration'
        :param agent: Specific agent identifier (usually maps to a class-level path attribute)
        :return: Instance of the specified agent class
        """
        if agent_type not in AgentFactory._registry:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        agent_class = AgentFactory._registry[agent_type]
        return agent_class.create(agent)


# ======================= ✅ Test Example ======================= #
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = AgentFactory.create_agent("evolution", "TriplesCompletionAgent")
        result = await agent.run(incomplete_triples="(OpenAI, ceo, )", type="Name")
        print(result)

    asyncio.run(test_agent())
