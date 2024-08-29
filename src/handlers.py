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
import tempfile 

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.config import bot_tg, is_processing
from src.utils import fetch_and_process_competitor_data, get_website_theme, check_wordpress
from src.openai_utils import filter_article_titles, check_article_exists
from src.serpstat import get_url_competitors, get_domain_keywords

dp = Dispatcher()

class Form(StatesGroup):
    waiting_for_message = State()

async def send_xlsx_to_chat(chat_id: int) -> None:
    # Используем временный файл
    file_path = os.path.join(tempfile.gettempdir(), 'output.xlsx')
    
    print("Temporary file path for sending:", file_path)
    
    # Проверяем, существует ли файл и не пустой ли он
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        document = FSInputFile(file_path)
        await bot_tg.send_document(chat_id, document)
    else:
        print("Ошибка: файл пустой или не существует.")
        raise ValueError("Файл пустой или не существует.")

async def save_df_to_xlsx(df: pd.DataFrame) -> None:
    # Используем временный файл
    file_path = os.path.join(tempfile.gettempdir(), 'output.xlsx')
    
    print("Temporary file path for saving:", file_path)
    
    df['validate title'] = np.nan

    # Сохранение DataFrame в формате Excel
    df.to_excel(file_path, index=False, engine='openpyxl')
    
    # Проверяем, что файл был создан и не пустой
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print("Файл успешно сохранен и не пустой.")
    else:
        print("Ошибка при сохранении файла.")
        raise ValueError("Ошибка при сохранении файла.")


@dp.message(Command('start'))
async def command_start_handler(message: Message, state: FSMContext) -> None:
    await message.answer(f"Приветствую тебя {message.from_user.first_name}!")
    await state.set_state(Form.waiting_for_message)

# Обработка текстовых сообщений
@dp.message(F.text.contains('http'))
async def handle_text_http_url(message: Message, state: FSMContext) -> None:
    global is_processing
    
    # Проверяем, идет ли уже обработка другой ссылки
    if is_processing:
        await message.answer("Обработка другой ссылки уже идет. Пожалуйста, подождите.")
        return
    
    # Устанавливаем флаг, что началась обработка
    is_processing = True
    
    try:
        async with aiohttp.ClientSession() as session:
            initial_url = message.text
            
            if await check_wordpress(initial_url, session) == True:
                list_of_topics = await fetch_and_process_competitor_data(message, session, initial_url, se='g_by', top_n=8)
                if not list_of_topics.empty:
                    await message.answer(f"Список предлагаемых тем представлен в файле.")
                    await save_df_to_xlsx(list_of_topics)
                    await send_xlsx_to_chat(message.chat.id)
            else:
                await message.answer("Представленный вами сайт определен, как не работающий на WordPress.")
        
    except Exception as e:
        logging.error(f"Ошибка в обработчике голосовых сообщений: {e}")
        await message.answer("Произошла ошибка при обработке вашего сообщения.")
    finally:
        is_processing = False

# Обработка текстовых сообщений
@dp.message(F.text)
async def handle_text_message(message: Message, state: FSMContext) -> None:
    try:
    
        await message.answer("Отправьте ссылку на сайт для анализа.")
       
    except Exception as e:
        logging.error(f"Ошибка в обработчике голосовых сообщений: {e}")
        await message.answer("Произошла ошибка при обработке вашего сообщения.")
    
def register_handlers1(dp: Dispatcher) -> None:
    dp.message.register(command_start_handler, CommandStart())
    dp.message.register(handle_text_http_url)
    # dp.message.register(handle_text_message)
    
