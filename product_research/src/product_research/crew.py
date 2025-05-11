from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

@CrewBase
class ProductResearch():
    """ProductResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def high_end_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['high_end_researcher'],
            verbose=True,
            max_rpm=5
        )

    @agent
    def budget_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['budget_researcher'],
            verbose=True,
            max_rpm=5
        )

    @agent
    def affordable_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['affordable_researcher'],
            verbose=True,
            max_rpm=5
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            max_rpm=5
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def high_end_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['high_end_research_task'],
            verbose=True,
            max_rpm=5
        )

    @task
    def budget_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['budget_research_task'],
            verbose=True,
            max_rpm=5
        )

    @task
    def affordable_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['affordable_research_task'],
            verbose=True,
            max_rpm=5
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ProductResearch crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
