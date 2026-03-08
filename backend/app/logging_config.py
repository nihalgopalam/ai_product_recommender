import logging
import sys

_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "pinecone", "langchain_core", "langgraph")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        filename='recommender.log',
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
