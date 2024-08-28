from pydantic_settings import BaseSettings

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

class Settings(BaseSettings):
    openai_api_key: str
    telegram_bot_token: str
    serpstat_api_key: str
    pinecone_api_key: str

    class Config:
        env_file = 'D:\Programming\Python\GPT\Wordpress_web_article_auto_generator\Wordpress_tests_bot\.env'
        
settings_pr= Settings()

bot_tg = Bot(token=settings_pr.telegram_bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# Создаем глобальный флаг состояния
is_processing = False

