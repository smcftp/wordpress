import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import time
import requests
import re
from urllib.parse import urlparse

from aiogram.types import Message

from sentence_transformers import SentenceTransformer, util

from src.serpstat import get_initial_domain_keywords, get_domain_keywords, get_task_result, get_task_status, add_task, get_competitors
from src.openai_utils import filter_article_titles, check_article_exists, filter_article_titles_theme
from src.emantic_similarity_comparison import process_semantic_similarity

# Настройка лимита семафора для ограничения числа одновременных запросов к OpenAI API
SEM_LIMIT = 5  # Максимальное число одновременных запросов
semaphore = asyncio.Semaphore(SEM_LIMIT)

# Получение тематики сайта
async def get_website_theme(url: str, session: aiohttp.ClientSession) -> str:
    try:
        
        parsed_url = urlparse(url)
        domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        
        async with session.get(domain_url) as response:
            response.raise_for_status() 

            # Получаем текст страницы
            html = await response.text()

            # Парсинг содержимого страницы
            soup = BeautifulSoup(html, 'html.parser')

            # Попытка найти описание сайта в метатегах
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description and meta_description.get('content'):
                input_str = str(meta_description.get('content'))
                cleaned_str = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', input_str)
                cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
                cleaned_str = cleaned_str.strip()
        
                return cleaned_str

            # Альтернативный поиск по содержимому страницы
            title = str(soup.title.string if soup.title else 'Без названия')
            cleaned_str = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', title)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
            cleaned_str = cleaned_str.strip()
            return cleaned_str

    except aiohttp.ClientError as e:
        return f"Ошибка при доступе к сайту: {e}"

async def process_url_limited(url: str) -> pd.DataFrame:
    """
    This function limits the number of concurrent requests to the API using a semaphore and 
    includes a delay to avoid hitting rate limits.

    Args:
    - url: The URL to check if it contains an article.

    Returns:
    - A DataFrame with the title and URL if the article exists, otherwise an empty DataFrame.
    """
    async with semaphore:  # Use the semaphore to limit concurrent requests
        await asyncio.sleep(0.2)  # Delay to prevent rate limiting
        
        # Check if the URL is an article
        article_exists = await check_article_exists(url)

        if article_exists:
            try:
                # Get the article title
                title = await get_article_title(url)
                
                # Create a new row with the article data
                new_row = pd.DataFrame({'title': [title], 'url': [url]})
                
                return new_row
            except Exception as e:
                print(f"Ошибка при получении заголовка статьи: {e}")
                return pd.DataFrame(columns=['title', 'url'])
        
        # Return an empty DataFrame if the article doesn't exist
        return pd.DataFrame(columns=['title', 'url'])

async def get_article_title(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                return soup.find('title').get_text() if soup.find('title') else "No Title Found"
    except aiohttp.ClientError as e:
        print(f"Ошибка при запросе: {e}")
        return None
    
###############################################################################################################

# async def get_competitor_keywords(initial_url, se='g_by', top_n=5):
#     start_time = time.time()
    
#     async with aiohttp.ClientSession() as session:
#         keywords_list = await get_initial_domain_keywords(session, initial_url, se)
#         keywords_list = keywords_list.sample(frac=1).reset_index(drop=True)
#         print("keywords list = ", keywords_list)
#         keywords = keywords_list.iloc[0, 0]
#         print("keywords = ", keywords)
        
#         # task_id = await add_task(session, keywords, 1, 1, 113, 2112, 1) # Беларусь
#         # task_id = await add_task(session, keywords, 1, 1, 20, 2643, 1) # Россия
#         task_id = await add_task(session, keywords, 1, 1, 23, 21176, 1) # СШA
        
#         print("task_id =", task_id)
        
#         while True:
#             await asyncio.sleep(20)
#             status = await get_task_status(session, task_id)
#             if status:
#                 break
#             else:
#                 print("Задача не выполнена. Попробуйте снова позже.")
        
#         if task_id:
#             top_urls = await get_task_result(session, task_id)
#             top_urls = top_urls.sort_values(by='position', ascending=True).head(top_n)
            
#             columns = ['title', 'url']
#             existing_df = pd.DataFrame(columns=columns)
            
#             print("Сайты конкуренты получены.")
#             tasks = []
#             for domain in top_urls['domain']:
#                 url_data = await get_domain_keywords(session, domain, se)
                
#                 if url_data is not None and not url_data.empty:
#                     url_data = url_data.head(15)
#                     for url in url_data['url']:
#                         tasks.append(process_url(session, url, existing_df))
            
#             results = await asyncio.gather(*tasks)
#             existing_df = pd.concat(results, ignore_index=True)
                               
#         # file_path = "output.csv"
#         # await save_dataframe_to_csv(existing_df, file_path)
#         # print("Итоговые темы:\n", existing_df)
#         existing_df = existing_df[['title']]
#         existing_df = existing_df.dropna()
        
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Время выполнения: {execution_time:.2f} секунд")
    
#     return existing_df     
    

# Функция для получения всех статей
async def fetch_all_posts(domain: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    
    parsed_url = urlparse(domain)
    domain = parsed_url.netloc
    
    url = f"https://{domain}/wp-json/wp/v2/posts"
    page = 1
    per_page = 100
    all_posts = []

    while True:
        async with session.get(f"{url}?per_page={per_page}&page={page}") as response:
            if response.status != 200:
                print(f"Failed to retrieve posts from page {page}, status code: {response.status}")
                break
            
            posts = await response.json()
            if not posts:
                break
            
            all_posts.extend(posts)
            page += 1

    df = pd.DataFrame(all_posts)
    
    # Текстовый запрос для фильтрации
    # query = "информативная и оригинальная статья, которая предоставляет полезную информацию, советы, рекомендации, исследовательский материал или новостные обзоры"
    
    query = "убери Новостные публикации, Рекламные и партнерские посты, Инфографики и визуальный контент"

    # Преобразование текстов в векторы
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query)
    title_embeddings = model.encode(df['title'].tolist())

    # Вычисление косинусного сходства
    cosine_scores = util.pytorch_cos_sim(query_embedding, title_embeddings)

    # Установка порога для фильтрации
    threshold = 0.5
    df['similarity'] = cosine_scores[0].cpu().numpy()

    # Фильтрация DataFrame на основе семантической схожести
    filtered_df = df[df['similarity'] >= threshold]

    # print("Отфильтрованные статьи на основе семантической схожести:")
    # print(filtered_df[['title', 'similarity']])
    
    # file_path = "C:\\Users\\thatn\\Downloads\\put3.xlsx"

    # # Сохранение DataFrame в формате Excel
    # filtered_df.to_excel(file_path, index=False, engine='openpyxl')

    return filtered_df

# Проверка администрируется ли сайт на World Press
async def check_wordpress(site_url: str, session: aiohttp.ClientSession) -> bool:
    parsed_url = urlparse(site_url)
    domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    
    paths = [
        '/wp-login.php',
        '/wp-admin/',
        '/wp-content/',
        '/wp-includes/',
        '/readme.html'
    ]
    
    async def check_path(path):
        url = domain_url.rstrip('/') + path
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return True
        except aiohttp.ClientError:
            pass
        return False
    
    # Check all paths asynchronously
    tasks = [check_path(path) for path in paths]
    results = await asyncio.gather(*tasks)
    
    if any(results):
        return True
    
    try:
        async with session.get(domain_url) as response:
            headers = response.headers
    except aiohttp.ClientError:
        return False

    if 'X-Powered-By' in headers and 'WordPress' in headers['X-Powered-By']:
        return True
    
    if 'X-Generator' in headers and 'WordPress' in headers['X-Generator']:
        return True
    
    return False
    
    
async def fetch_and_process_competitor_data(message: Message, session: aiohttp.ClientSession, initial_url: str, se='g_by', top_n=5) -> pd.DataFrame:
    start_time = time.time()
    
    try:
        # Получение конкурентов через "Конкуренты домена"
        try:
            top_urls = await get_competitors(session, initial_url, se)
            if top_urls.empty:
                await message.answer(f"Конкуренты для данного сайта не найдены.") 
                return pd.DataFrame() 
            formatted_output = top_urls.to_string(index=False, justify='left')
            await message.answer(f"Сайты конкуренты получены.\n```\n{formatted_output}\n```", parse_mode="Markdown")
        except Exception as e:
            await message.answer(f"Ошибка при получении конкурентов: {e}")
            return pd.DataFrame()  

        # Настройка размера массива конкурентов
        percentage = 80  # Процент, например, 50 для 50%
        num_rows = int(len(top_urls) * (percentage / 100))
        # url_data = top_urls.head()
                
        columns = ['title', 'url']
        existing_df = pd.DataFrame(columns=columns)
        all_url_data = pd.DataFrame()
        
        tasks = []
        for domain in top_urls.get('domain', []):
            try:
                url_data = await get_domain_keywords(session, domain, se)
            except Exception as e:
                print(f"Ошибка при получении ключевых слов для домена {domain}: {e}")
                continue  # Переходим к следующему домену в случае ошибки
            
            if url_data is not None and not url_data.empty:
                all_url_data = pd.concat([all_url_data, url_data], ignore_index=True)
                
                # Настройка размера массива
                percentage = 10  # Процент, например, 50 для 50%
                num_rows = int(len(url_data) * (percentage / 100))
                url_data = url_data.head(50)
                
                for url in url_data['url']:
                    tasks.append(process_url_limited(url))
        
        # Обработка данных ссылок на статьи
        try:
            all_url_data.dropna(inplace=True)
            all_url_data.drop_duplicates(subset='url', keep='first', inplace=True)
            all_url_data.sort_values(by=['found_results', 'position', 'traff'], ascending=[False, False, False], inplace=True)
            print("all_url_data = \n", all_url_data)
            print(all_url_data.info())
        except Exception as e:
            print(f"Ошибка при обработке данных ссылок на статьи: {e}")
        
        try:
            results = await asyncio.gather(*tasks)
            existing_df = pd.concat(results, ignore_index=True)
            existing_df.dropna(inplace=True)
            existing_df.drop_duplicates(subset='title', keep='first', inplace=True)
            print("existing_df = \n", existing_df)
            print(existing_df.info())
        except Exception as e:
            print(f"Ошибка при обработке существующих данных: {e}")
            return pd.DataFrame()

        # Фильтрация данных через LLM: дубликаты и мусор
        print("Проверка LLM начата")
        try:
            df_filtered = pd.DataFrame(await filter_article_titles(existing_df), columns=['title'])
            filtered_titles = set(df_filtered['title'])
            existing_df = existing_df[existing_df['title'].isin(filtered_titles)]
            print("Проверка LLM закончена")
        except Exception as e:
            print(f"Ошибка при фильтрации данных через LLM: {e}")
        
        # Получение существующих статей на сайте
        try:
            cur_articles_df = await fetch_all_posts(initial_url, session)
            print("Сбор имеющихся статей завершен")
        except Exception as e:
            print(f"Ошибка при сборе имеющихся статей: {e}")
            return pd.DataFrame()

        # Фильтрация статей по семантическому сходству
        similarity_threshold = 0.8
        try:
            filtered_df = await process_semantic_similarity(existing_df, cur_articles_df, similarity_threshold)
            print("Фильтрация статей по семантическому сходству завершена")
        except Exception as e:
            print(f"Ошибка при фильтрации статей по семантическому сходству: {e}")
            return pd.DataFrame()

        # Соединение двух дата фреймов по ссылке
        try:
            merged_df = pd.merge(filtered_df, all_url_data[['url', 'found_results', 'position', 'traff']], on='url', how='left')
            merged_df.dropna(inplace=True)
            merged_df.drop_duplicates(inplace=True)
            merged_df.sort_values(by=['found_results', 'position', 'traff'], ascending=[False, False, False], inplace=True)
            print("\n\n\n\nИтог:\n", merged_df)
        except Exception as e:
            print(f"Ошибка при объединении данных: {e}")
            return pd.DataFrame()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Время выполнения: {execution_time:.2f} секунд")
        
        return merged_df
    
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return pd.DataFrame()


async def pars_tops(session, initial_url, se, top_n):
    
    keywords_list = await get_initial_domain_keywords(session, initial_url, se)
    keywords_list = keywords_list.sample(frac=1).reset_index(drop=True)
    print("keywords list = ", keywords_list)
    keywords = keywords_list.iloc[0, 0]
    print("keywords = ", keywords)
        
    # task_id = await add_task(session, keywords, 1, 1, 113, 2112, 1) # Беларусь
    # task_id = await add_task(session, keywords, 1, 1, 20, 2643, 1) # Россия
    task_id = await add_task(session, keywords, 1, 1, 23, 21176, 1) # СШA
        
    print("task_id =", task_id)
        
    while True:
        await asyncio.sleep(20)
        status = await get_task_status(session, task_id)
        if status:
            break
        else:
            print("Задача не выполнена. Попробуйте снова позже.")
        
    if task_id:
        top_urls = await get_task_result(session, task_id)
        top_urls = top_urls.sort_values(by='position', ascending=True).head(top_n)
        return top_urls


# async def main() -> None:
#     # start_time = time.time()
#     async with aiohttp.ClientSession() as session:
#         initial_url = 'https://ppc-rating.by/'
#         message = None
#         await fetch_and_process_competitor_data(message, session, initial_url, se='g_by', top_n=5)

# asyncio.run(main())
