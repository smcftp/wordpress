import logging
import aiohttp
import asyncio

from aiogram import Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from aiogram.types.input_file import FSInputFile
import pandas as pd
import numpy as np

import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.config import bot_tg, is_processing
import src.db_utils

dp = Dispatcher()

# Классы состояний
class Form(StatesGroup):
    waiting_for_message = State()

class PublishIntervalState(StatesGroup):
    waiting_for_interval = State()

# Обработчик команды /start
@dp.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    await message.answer(f"Приветствую тебя {message.from_user.first_name}!")
    await state.set_state(Form.waiting_for_message)

# Обработка сообщений с URL
@dp.message(F.text.contains('http'))
async def handle_text_http_url(message: Message, state: FSMContext) -> None:
    global is_processing

    if is_processing:
        await message.answer("Обработка другой ссылки уже идет. Пожалуйста, подождите.")
        return

    # Сохраняем ссылку в FSM
    await state.update_data(initial_url=message.text)

    # Спрашиваем интервал публикации
    await message.answer("Введите интервал публикации в часах (например, 72 часа для 3 дней). Минимальный интервал публикации - 1 час:")
    await state.set_state(PublishIntervalState.waiting_for_interval)  # Изменено на правильный метод

# Обработка введённого интервала публикации
@dp.message(PublishIntervalState.waiting_for_interval)
async def process_publish_interval(message: Message, state: FSMContext) -> None:
    global is_processing

    try:
        interval_text = message.text
        # Проверяем, что введённое значение является числом и находится в пределах от 1 до 749
        if not interval_text.isdigit() or not (1 <= int(interval_text) < 750):
            await message.answer("Некорректный интервал публикации. Пожалуйста, введите число от 1 до 749.")
            return

        publication_interval = int(interval_text)
        await state.update_data(publication_interval=publication_interval)

        is_processing = True

        # Извлекаем данные из FSM
        user_data = await state.get_data()
        initial_url = user_data.get('initial_url')

        # Логика обработки
        async with aiohttp.ClientSession() as session:
            await message.answer(f"Обработка завершена с интервалом публикации: {publication_interval} часов.")
            await db_utils.periodic_article_processing(message, session, initial_url, message.from_user.id, message.from_user.username, publication_interval)
            # await db_utils.fjfj(message)

    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer("Произошла ошибка при обработке вашего сообщения.")
    finally:
        is_processing = False
        await state.clear()  # Очищаем состояние

# Обработка других текстовых сообщений
@dp.message(F.text)
async def handle_text_message(message: Message, state: FSMContext) -> None:
    try:
        await message.answer("Отправьте ссылку на сайт для анализа.")
    except Exception as e:
        logging.error(f"Ошибка в обработчике сообщений: {e}")
        await message.answer("Произошла ошибка при обработке вашего сообщения.")

# Регистрация хендлеров
def register_handlers1(dp: Dispatcher) -> None:
    dp.message.register(command_start_handler, CommandStart())
    dp.message.register(handle_text_http_url, F.text.contains("http"))
    dp.message.register(process_publish_interval, PublishIntervalState.waiting_for_interval)
    dp.message.register(handle_text_message, F.text)
    
