import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from autogen_selector.hosting import container
from autogen_selector.protocols.i_azure_openai_service import IAzureOpenAIService
from autogen_selector.tools.tools import (
    get_bank_account_id,
    get_investment_account_balance,
    get_saving_account_balance,
)

llm_client = container[IAzureOpenAIService].get_model()

customer_agent = AssistantAgent(
    "customer_agent",
    model_client=llm_client,
    description="A bank assistant.",
    system_message="""You are a bank assistant.
    Your job is to break down complex tasks into smaller, manageable subtasks.

    Your team members are:
        account_agent: provides account ID
        saving_account_agent: provides saving account balance
        investment_agent: provides investment account balance

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    <agent> : <task>

    After all tasks are complete,  Provide your response in a JSON format."

    ```json{
        "account_id": "<account id>",
        "saving_balance": <saving balance>,
        "investment_balance": <investment balance>,
        "total_balance": <total balance>
    }```

    And, end with "TERMINATE".
    """,
)

account_agent = AssistantAgent(
    "account_agent",
    model_client=llm_client,
    description="An account agent.",
    tools=[get_bank_account_id],
    system_message="""You are an account agent who can provide account ID.
You should always use the tool provided to generate the account balance.
    """,  # noqa E501
)

investment_agent = AssistantAgent(
    "investment_agent",
    model_client=llm_client,
    description="An investment account agent.",
    tools=[get_investment_account_balance],
    system_message="""You are an investment account agent who can provide information about the investment account balance.
Look at the chat history to understand the context of the conversation and the account id is in the it. Look for nvestment account ID.
You should always use the tool provided to generate the account balance.
    """,  # noqa E501
)

saving_account_agent = AssistantAgent(
    "saving_account_agent",
    model_client=llm_client,
    description="A saving account agent.",
    tools=[get_saving_account_balance],
    system_message="""You are a saving account agent who can provide information about the saving account balance.
Look at the chat history to understand the context of the conversation and the account id is in the it. Look for Saving account ID.
You should always use the tool provided to generate the account balance.
    """,  # noqa E501
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the
next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

team = SelectorGroupChat(
    [customer_agent, account_agent, saving_account_agent, investment_agent],
    model_client=llm_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)


async def main():
    await Console(
        team.run_stream(
            task="Get the account ID and then get the saving balance "
            "and investment balance. Both saving and investment account have"
            "the same account ID. Sum the balances when they are available."
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
