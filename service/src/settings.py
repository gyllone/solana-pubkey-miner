
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LIP_PATH: str = "lib/lib-pubkey-miner.so"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8101


settings = Settings()
