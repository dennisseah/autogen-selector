from dataclasses import dataclass

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from lagom.environment import Env

from autogen_selector.protocols.i_azure_openai_service import IAzureOpenAIService


class AzureOpenAIServiceEnv(Env):
    azure_openai_endpoint: str
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureOpenAIService(IAzureOpenAIService):
    env: AzureOpenAIServiceEnv

    def get_model(self) -> AzureOpenAIChatCompletionClient:
        if self.env.azure_openai_api_key:
            return AzureOpenAIChatCompletionClient(
                azure_endpoint=self.env.azure_openai_endpoint,
                api_key=self.env.azure_openai_api_key,
                model=self.env.azure_openai_deployed_model_name,
                api_version=self.env.azure_openai_api_version,
            )

        azure_ad_token = (
            DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default")
            .token
        )

        return AzureOpenAIChatCompletionClient(
            azure_endpoint=self.env.azure_openai_endpoint,
            azure_ad_token=azure_ad_token,
            model=self.env.azure_openai_deployed_model_name,
            api_version=self.env.azure_openai_api_version,
        )
