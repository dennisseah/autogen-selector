from typing import Protocol

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class IAzureOpenAIService(Protocol):
    def get_model(self) -> AzureOpenAIChatCompletionClient:
        """
        Return the Azure OpenAI model with the given temperature.

        The parameters for constructing the model are read from the environment
        variables.

        :return: The Azure OpenAI model with the given temperature
        """
        ...
