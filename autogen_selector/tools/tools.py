import logging
import random

from autogen_selector.hosting import container

logger = container[logging.Logger]


def get_bank_account_id() -> str:
    """Get the bank account ID."""
    id = random.choice(["123", "456", "789"])
    logger.info(f"[TOOL] Bank account ID: {id}")
    return id


def get_investment_account_balance(bank_account_id: str) -> float:
    """Get the investment balance of an account.

    :param bank_account_id: The account ID.
    :return: The investment balance.
    """
    balance = {
        "123": 100000.10,
        "456": 200000.20,
        "789": 300000.30,
    }[bank_account_id]
    logger.info(f"[TOOL] Investment balance: {balance}")
    return balance


def get_saving_account_balance(bank_account_id: str) -> float:
    """Get the saving balance of an account.

    :param bank_account_id: The account ID.
    :return: The saving balance.
    """
    balance = {
        "123": 10000.10,
        "456": 20000.20,
        "789": 30000.30,
    }[bank_account_id]
    logger.info(f"[TOOL] Saving balance: {balance}")
    return balance
