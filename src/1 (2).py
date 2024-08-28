import logging
import tempfile
import asyncio
import os
import json
from typing import Optional

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram import Dispatcher, F

from aiogram import Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from typing_extensions import override

from models import User, UserValue
from database import SessionLocal

TOKEN = '7008514593:AAEu4izBjQhOsbBdRQK8XgA1RwnKTH5ECeI'

# Создание бота и диспетчера
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def save_user_value(telegram_id: int, value: str):
    logger.debug(f"Сохранение ценности для telegram_id={telegram_id}, value={value}")
    try:
        async with SessionLocal() as session:
            async with session.begin():
                logger.debug("Начало транзакции")
                user = await session.execute(select(User).filter_by(telegram_id=telegram_id))
                user = user.scalars().first()
                if not user:
                    logger.debug("Пользователь не найден, создается новый пользователь")
                    user = User(telegram_id=telegram_id)
                    session.add(user)
                    await session.commit()
                logger.debug("Добавление новой ценности")
                user_value = UserValue(user_id=user.id, value=value)
                session.add(user_value)
                await session.commit()
                logger.debug("Ценность успешно сохранена")
    except Exception as e:
        logger.error(f"Ошибка при сохранении ценности для telegram_id={telegram_id}: {str(e)}")
        raise

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        telegram_id = message.from_user.id
        value = 'Любовь к птицам'
        
        await save_user_value(telegram_id, value)
    except Exception as e:
        logging.error(f"Ошибка в обработчике команды start: {e}")

async def main() -> None:
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Ошибка в основной функции: {e}")

if __name__ == "__main__":
    asyncio.run(main())