from sqlalchemy.orm import Session
import pandas as pd
import asyncio
import aiohttp
import os

from aiogram.types import Message
from aiogram.types.input_file import FSInputFile

from urllib.parse import urlparse

from bcrypt import gensalt, hashpw
import src.utils as utils

from src.config import bot_tg
from src.db.database import SessionLocal
import src.db.crud as crud

# Функция для сохранения данных в базе данных.
async def save_data_to_db(df: pd.DataFrame, initial_url: str, user_id: int, user_username: str, publication_interval: int):
    
    # Создание новой сессии базы данных
    db_session = SessionLocal()
    
    # Данные о пользователе и сайте
    password = "user_password"
    hashed_password = hashpw(password.encode('utf-8'), gensalt())

    user_data = {
        "user_id": user_id,
        "username": user_username,
        "email": str(user_username + "@example.com"),
        "password_hash": str(hashed_password),
    }
    
    site = await crud.get_site_by_user_and_url(db_session, user_id, initial_url)
    
    parsed_url = urlparse(initial_url)
    domain = parsed_url.netloc
    
    if site:
        site_id = site.id
        
        site_data = {
            "site_id": site_id,
            "user_id": user_data["user_id"],
            "site_url": initial_url,
            "site_name": domain, 
            "publication_interval": publication_interval
        }
        
        article_id = await crud.get_last_article_id(db_session)
        
        if article_id != None:

            # Преобразование данных из DataFrame в список статей
            article_data = df.to_dict('records')

            # Добавление article_id и site_id в данные о статьях
            for i, article in enumerate(article_data, start=article_id+1):
                article["article_id"] = i
                article["site_id"] = site_data["site_id"]
        
    else:
        
        site_id = await crud.get_last_site_id(db_session)
        
        if site_id != None:
        
            site_data = {
            "site_id": site_id+1,
            "user_id": user_data["user_id"],
            "site_url": initial_url,
            "site_name": domain, 
            "publication_interval": publication_interval
        }

            article_id = await crud.get_last_article_id(db_session)
        
            if article_id != None:

                # Преобразование данных из DataFrame в список статей
                article_data = df.to_dict('records')

                # Добавление article_id и site_id в данные о статьях
                for i, article in enumerate(article_data, start=article_id+1):
                    article["article_id"] = i
                    article["site_id"] = site_data["site_id"]
    
    # Очистка БД
    # try:
    #     await clear_database(db)
    #     print("База данных успешно очищена.")
    # except Exception as e:
    #     print(f"Ошибка при очистке базы данных: {e}")
    # finally:
    #     db.close()


    # Попробуем найти пользователя по email или имени
    user = await crud.get_user_by_id(db_session, user_data["user_id"])

    try:
        # Проверка, существует ли пользователь
        if user:
            # Пользователь существует, проверем список сайтов
            print(f"Пользователь уже существует: {user.username} ({user.email})")
            
            site = await crud.get_site_by_user_and_url(db_session, user_id=user.id, site_url=site_data['site_url'])

            if site:
                # Сайт существует, проверем список статей
                print(f"Сайт уже существует: {site.site_url} ({site.site_name})")
                
                # Получение всех статей, связанных с сайтом
                existing_articles = await crud.get_articles_by_site_id(db_session, site_id=site.id)
                existing_article_titles = {article.article_title for article in existing_articles}
                
                for article in article_data:
                    title = article['title']
                    if title in existing_article_titles:
                        # Статья существует
                        print(f"Статья уже существует: {title}")
                    else:
                        # Статья не существует, добавляем ее
                        print(f"Статья не существует: {title}")
                        new_article = await crud.create_article(db_session, article_id=article["article_id"], site_id=site.id, article_title=article['title'], article_url=article['url'])
                        print(f"Добавлена статья: {new_article.article_title}")
                
            else:
                # Сайта не существует, добавление нового сайта
                site = await crud.create_site(db_session, site_id=site_data["site_id"], user_id=user.id, site_url=site_data['site_url'], site_name=site_data['site_name'], publication_interval=site_data["publication_interval"])
                print(f"Добавлен сайт: {site.site_url}")
                
                # Добавление статей
                for article in article_data:
                    new_article = await crud.create_article(db_session, article_id=article["article_id"], site_id=site.id, article_title=article['title'], article_url=article['url'])
                    print(f"Добавлена статья: {new_article.article_title}")
            
        else:
            # Пользователя нет, создание нового пользователя
            user = await crud.create_user(db_session, user_id=user_data["user_id"], username=user_data['username'], email=user_data['email'], password_hash=user_data['password_hash'])
            print(f"Создан пользователь: {user.username} ({user.email})")
            
            # Добавление сайта
            site = await crud.create_site(db_session, site_id=site_data["site_id"], user_id=user.id, site_url=site_data['site_url'], site_name=site_data['site_name'], publication_interval=site_data["publication_interval"])
            print(f"Добавлен сайт: {site.site_url}")

            # Добавление статей
            for article in article_data:
                new_article = await crud.create_article(db_session, article_id=article["article_id"], site_id=site.id, article_title=article['title'], article_url=article['url'])
                print(f"Добавлена статья: {new_article.article_title}")

    except Exception as e:
        print(f"Ошибка при сохранении данных в БД: {e}")

    finally:
        # Закрытие сессии базы данных
        db_session.close()


# Получение последней темы для статей из бд
# async def periodic_article_processing():
async def periodic_article_processing(message: Message, session: aiohttp.ClientSession, initial_url: str, user_id: int, user_username: str, publication_interval):
    
    # Создание новой сессии базы данных
    db_session = SessionLocal()
    
    a = 1
    
    while a == 1:
        
        user = await crud.get_user_by_id(db_session, user_id)
        # se = "g_kz"
        # se = "g_ua"
        se = "g_by"
        # se = "g_us"
        # se = "g_bg"

        if user:
            site = await crud.get_site_by_user_and_url(db_session, user_id, initial_url)
            if site:
                site_id = site.id
                article = await crud.get_and_delete_first_article_by_site_id(db_session, site_id)
                if article:
                    
                    print("\n\nНапсиание статьи!\n\n")
                    
                    # Название статьи найдено, пишем статью
                    article_title = article.article_title  # Извлекаем значение title
                    url = article.article_url  # Извлекаем значение url
                  
                    session = None
                    
                    # from Article_generation.dev.dev_Article_gen_initial_text_through_plan import gen_keyword_article
                    from src.Article_generation.dev.test_1_Article_gen_initial_text_through_plan import gen_keyword_article
                    from src.Article_generation.dev.img_imput import add_img_to_textarticle
                    
                    key_words_article, keywords_article_title = await gen_keyword_article(url, article_title, session)
                    
                    img_gen = True
                    
                    file_path = await add_img_to_textarticle(key_words_article, keywords_article_title, img_gen)
                    document = FSInputFile(file_path)
                    await bot_tg.send_document(message.chat.id, document)

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, os.remove, file_path)
                    
                else:
                    title_df = await utils.fetch_and_process_competitor_data(message, session, initial_url, se=se, top_n=10)
                    await save_data_to_db(title_df, initial_url, user_id, user_username, publication_interval)
            else:
                title_df = await utils.fetch_and_process_competitor_data(message, session, initial_url, se=se, top_n=10)
                await save_data_to_db(title_df, initial_url, user_id, user_username, publication_interval)
        else:
            title_df = await utils.fetch_and_process_competitor_data(message, session, initial_url, se=se, top_n=10)
            await save_data_to_db(title_df, initial_url, user_id, user_username, publication_interval)
            
        a = 2


