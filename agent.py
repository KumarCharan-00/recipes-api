import asyncio
import os
from pydoc import describe

import dotenv
from github import Github, Auth

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent, FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI

dotenv.load_dotenv()

token = os.getenv("GITHUB_TOKEN")
git = Github(auth=Auth.Token(token)) if token else None

repo_url = "https://github.com/KumarCharan-00/recipes-api.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

if git is not None:
    repo = git.get_repo(full_repo_name)
else:
    repo = None

def get_pr_details(pr_number: int) -> dict:
    """Gets details about a given pull request."""
    pr = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pr.get_commits()]
    return {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha,
        "commit_SHAs": commit_SHAs
    }

def get_file_contents(file_path: str) -> str:
    """Gets the contents of a file from the repository given a file path."""
    return repo.get_contents(file_path).decoded_content.decode('utf-8')

def get_pr_commit_details(commit_sha: str) -> list:
    """Gets details about a commit given the commit SHA."""
    commit = repo.get_commit(commit_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

# Context Agent
## Convert functions to tools
pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
files_tool = FunctionTool.from_defaults(fn=get_file_contents)
pr_commits_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)

async def gathered_context(ctx: Context, gathered_contexts: str) -> str:
    """Adds the gathered context to the state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "Gathered context added to state successfully"

add_context_to_state = FunctionTool.from_defaults(async_fn=gathered_context)

llm = GoogleGenAI(model="gemini-3.1-flash-lite-preview")

context_system_prompt = """You are the context gathering agent. When gathering context, you MUST gather \n: 
    - The details or contents: author, title, body, diff_url, state, and head_sha; \n
    - Changed files; \n
    - Any requested for files; 
    Once you gather the requested info, you MUST hand control back to the Commentor Agent.
    \n"""

context_agent = FunctionAgent(
    tools=[pr_details_tool, files_tool, pr_commits_details_tool, add_context_to_state],
    llm=llm,
    system_prompt=context_system_prompt,
    name="ContextAgent",
    description="Gather PR context and details from GitHub to pass to the CommentorAgent.",
    can_handoff_to = ["CommentorAgent"]
)

# Commentor Agent

async def draft_comment_state(ctx: Context, draft_comment: str) -> str:
    """Adds the drafted comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment added to state successfully"

add_comment_to_state = FunctionTool.from_defaults(async_fn=draft_comment_state)

commentor_system_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
        Ensure to do the following for a thorough review:
         - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
         - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
            - What is good about the PR? \n
            - Did the author follow ALL contribution rules? What is missing? \n
            - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
            - Are new endpoints documented? - use the diff to determine this. \n
            - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
         - If you need any additional details, you must hand off to the context Agent. \n
         - You should directly address the author. So your comments should sound like: \n
         - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n
         'Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?' """

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state],
    system_prompt=commentor_system_prompt,
    can_handoff_to = ["ContextAgent", "ReviewAndPostingAgent"]
)

# Review and Posting Agent

async def review_comment_state(ctx: Context, review_comment: str) -> str:
    """Adds the drafted comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["review_comment"] = review_comment
    await ctx.store.set("state", current_state)
    return "Review comment added to state successfully"

final_review = FunctionTool.from_defaults(async_fn=review_comment_state)

async def add_comment_gh(ctx: Context, pr_number: int) -> str:
    """posts comment in github under specified PR number"""
    pr = repo.get_pull(number=pr_number)
    state = await ctx.store.get("state")
    pr.create_review(body=state["review_comment"])
    return "Comment posted in github Successfully"

post_comment = FunctionTool.from_defaults(async_fn=add_comment_gh)

review_n_posting_sys_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub."""

review_n_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Uses the Review and Posting agent to create a review comment.",
    tools=[final_review, post_comment],
    system_prompt=review_n_posting_sys_prompt,
    can_handoff_to = ["CommentorAgent"]
)

# WorkFlow Agent

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_n_posting_agent],
    root_agent=commentor_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "review_comment": ""
    },
)


async def main():
    pr_number = os.getenv("PR_NUMBER")
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

if __name__ == "__main__":
    asyncio.run(main())
    git.close()