import random
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEYS: str
    SPIDER_API_KEY: str
    BASE_URL: str = ""
    MODEL: str = ""
    EMBEDDING_MODEL: str = ""
    SCHEMA_EXPLORE_BATCH_SIZE: int = 15

    @property
    def API_KEY(self) -> str:
        """Randomly pick one OpenAI API key from the list."""
        keys = [k.strip() for k in self.OPENAI_API_KEYS.split(",") if k.strip()]
        if not keys:
            raise ValueError("OPENAI_API_KEYS is empty!")
        return random.choice(keys)

    class Config:
        env_file = "../../.env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Retrieve and cache settings."""
    return Settings()

settings = get_settings()
