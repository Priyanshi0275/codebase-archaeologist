"""
Three CrewAI agents investigate a question about the codebase, each
grounded by retrieval from Chroma (the commit history we ingested):

1. Historian        — why does this code exist?
2. Dependency Mapper — what's risky about changing it?
3. Refactor Drafter  — concrete suggestion, written in the repo's own tone

Sequential process: each agent's output is passed as context to the next.
"""
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.vectorstore import query as chroma_query
from app import style_service


@tool("search_commit_history")
def search_commit_history(question: str) -> str:
    """Search the ingested git commit history for context relevant to
    the given question or function/file name. Returns the most relevant
    commit messages, diffs, and metadata."""
    results = chroma_query(question, n_results=5)
    if not results:
        return "No relevant commit history found. Has the repo been ingested yet?"
    formatted = []
    for r in results:
        meta = r["metadata"]
        formatted.append(
            f"[commit {meta.get('sha')} by {meta.get('author')} on {meta.get('date')}]\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(formatted)


def _llm() -> LLM:
    return LLM(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)


def build_crew(question: str) -> Crew:
    llm = _llm()

    historian = Agent(
        role="Git Historian",
        goal="Explain WHY a piece of code exists, grounded in commit history evidence.",
        backstory=(
            "You are a meticulous software archaeologist. You never guess — "
            "you search commit history for evidence before making a claim, "
            "and you cite the commit SHA you're relying on."
        ),
        tools=[search_commit_history],
        llm=llm,
        verbose=True,
    )

    dependency_mapper = Agent(
        role="Dependency Risk Analyst",
        goal="Identify what else in the codebase might depend on or be affected by this code, based on commit history patterns (files frequently changed together, related fixes).",
        backstory=(
            "You've seen too many refactors break unrelated features. You look "
            "for evidence in commit history of what tends to change alongside "
            "the code in question, and flag risk level (low/medium/high) with reasoning."
        ),
        tools=[search_commit_history],
        llm=llm,
        verbose=True,
    )

    refactor_drafter = Agent(
        role="Refactor Advisor",
        goal="Propose a concrete, low-risk refactor or change suggestion that fits the codebase's own conventions.",
        backstory=(
            "You write suggestions that sound like they came from a senior "
            "engineer on this specific team — grounded, specific, never generic "
            "textbook advice."
        ),
        tools=[search_commit_history],
        llm=llm,
        verbose=True,
    )

    task1 = Task(
        description=(
            f"Investigate this question about the codebase: '{question}'\n"
            "Search commit history and explain why this code likely exists, "
            "citing specific commit SHAs as evidence."
        ),
        expected_output="A short, evidence-backed explanation (3-6 sentences) citing commit SHAs.",
        agent=historian,
    )

    task2 = Task(
        description=(
            f"Given the historian's findings about '{question}', identify what "
            "depends on this code or has broken alongside it before. Rate the "
            "risk of changing it as low/medium/high with reasoning."
        ),
        expected_output="A risk rating (low/medium/high) plus 2-4 sentences of reasoning, citing evidence where available.",
        agent=dependency_mapper,
        context=[task1],
    )

    task3 = Task(
        description=(
            f"Given the historian's and dependency analyst's findings about "
            f"'{question}', propose one concrete, low-risk next step or refactor "
            "suggestion. Keep it specific and actionable, matching the tone of "
            "this codebase's own commit messages."
        ),
        expected_output="A short, actionable suggestion (3-5 sentences), no generic advice.",
        agent=refactor_drafter,
        context=[task1, task2],
    )

    return Crew(
        agents=[historian, dependency_mapper, refactor_drafter],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )


def run_investigation(question: str) -> dict:
    crew = build_crew(question)
    result = crew.kickoff()

    refactor_raw = str(crew.tasks[2].output.raw) if crew.tasks[2].output else ""

    # Live LoRA styling: rewrite the refactor suggestion in the repo's own
    # commit-message voice, if an adapter is configured and loaded. Falls
    # back to the raw suggestion untouched if not — see style_service.py.
    refactor_styled = style_service.rewrite_in_style(refactor_raw) if refactor_raw else refactor_raw

    return {
        "question": question,
        "historian": str(crew.tasks[0].output.raw) if crew.tasks[0].output else "",
        "dependency_mapper": str(crew.tasks[1].output.raw) if crew.tasks[1].output else "",
        "refactor_drafter": refactor_raw,
        "refactor_drafter_styled": refactor_styled,
        "lora_style_applied": style_service.is_available(),
        "final_answer": str(result),
    }
