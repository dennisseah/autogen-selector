from pytest_mock import MockerFixture

from autogen_selector.services.azure_openai_service import (
    AzureOpenAIService,
    AzureOpenAIServiceEnv,
)


def test_get_model_with_key(mocker: MockerFixture):
    mocked_chat_openai = mocker.patch(
        "autogen_selector.services.azure_openai_service.AzureOpenAIChatCompletionClient"
    )
    svc = AzureOpenAIService(
        env=AzureOpenAIServiceEnv(
            azure_openai_endpoint="https://api.openai.com",
            azure_openai_api_key="test_key",
            azure_openai_api_version="2020-08-04",
            azure_openai_deployed_model_name="test_model",
        )
    )
    assert svc.get_model() is not None
    assert mocked_chat_openai.call_count == 1


def test_get_model_without_key(mocker: MockerFixture):
    mocked_chat_openai = mocker.patch(
        "autogen_selector.services.azure_openai_service.AzureOpenAIChatCompletionClient"
    )
    mocked_default_azure_credential = mocker.patch(
        "autogen_selector.services.azure_openai_service.DefaultAzureCredential"
    )
    svc = AzureOpenAIService(
        env=AzureOpenAIServiceEnv(
            azure_openai_endpoint="https://api.openai.com",
            azure_openai_api_version="2020-08-04",
            azure_openai_deployed_model_name="test_model",
        )
    )
    assert svc.get_model() is not None
    assert mocked_chat_openai.call_count == 1
    assert mocked_default_azure_credential.call_count == 1
