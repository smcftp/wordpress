import aiohttp
import pandas as pd
import json

from urllib.parse import urlparse
import asyncio

from config import settings_pr
import numpy as np

from aiogram.types import Message


SERPSTAT_API_TOKEN = settings_pr.serpstat_api_key

# URL для запросов к API Serpstat
url_pars = f'https://serpstat.com/rt/api/v2/?token={SERPSTAT_API_TOKEN}'
url_main = f'https://api.serpstat.com/v4/?token={SERPSTAT_API_TOKEN}'

# Перевод ключевых слов на английский
async def translate_to_english(session, text):
    try:
        url = 'https://translate.googleapis.com/translate_a/single'
        params = {
            'client': 'gtx',
            'sl': 'ru',
            'tl': 'en',
            'dt': 't',
            'q': text
        }
        async with session.get(url, params=params) as response:
            result = await response.json()
            return result[0][0][0]  # Получаем переведенный текст из JSON ответа
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

# Получение ключевых слов для парсинга топа
async def get_initial_domain_keywords(message: Message, session: aiohttp.ClientSession, url: str, se: str) -> pd.DataFrame:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    data = {
        "id": "1",
        "method": "SerpstatDomainProcedure.getDomainKeywords",
        "params": {
            "domain": str(domain),
            "se": str(se)
        }
    }
    try:
        async with session.post(url_main, json=data) as response:
            if response.status == 200:
                response_data = await response.json()
                if isinstance(response_data, dict) and 'result' in response_data:
                    response_data = response_data['result']['data']

                    tasks = [translate_to_english(session, item['keyword']) for item in response_data if 'keyword' in item]
                    translated_keywords = await asyncio.gather(*tasks)

                    return pd.DataFrame(translated_keywords)
            else:
                print(f"Error in get_initial_domain_keywords: {response.status}")
                return None
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

# Добавление задачи на парсинг сайта
async def add_task(message: Message, session: aiohttp.ClientSession, keywords, type_id, se_id, country_id, region_id, lang_id) -> int:
    payload = {
        "id": "some_id",
        "method": "tasks.addTask",
        "params": {
            "keywords": keywords,
            "typeId": type_id,
            "seId": se_id,
            "countryId": country_id,
            "regionId": region_id,
            "langId": lang_id
        }
    }
    try:
        async with session.post(url_pars, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, dict) and 'result' in result:
                    task_id = int(result['result']['task_id'])
                    return task_id
            else:
                print(f"Error in add_task: {response.status}")
                return None
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

# Получение статуса парсера
async def get_task_status(message: Message, session: aiohttp.ClientSession, type_id):
    
    payload = {
        "id": 1,
        "method": "tasks.getList",
        "params": {
            "page": 1,
            "pageSize": 1000
        }
    }
    
    try:
        async with session.post(url_pars, json=payload, timeout=10) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, dict) and 'result' in result:
                    for result in result['result']:
                        if int(result['task_id']) == int(type_id):
                            cur_ob = str(result['progress'])
                            cur_ob = cur_ob.replace(" ", "")
                            if cur_ob == "100%":
                                return True
            else:
                print(f"Error in add_task: {response.status}")
                return None
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

# Получение результатов парсинга сайта
async def get_task_result(message: Message, session, task_id):
    payload = {
        "id": "some_id",
        "method": "tasks.getTaskResult",
        "params": {
            "taskId": task_id
        }
    }
    try:
        async with session.post(url_pars, json=payload) as response:
            if response.status == 200:
                response_data = await response.json()
                
                tops_info = []
                
                if isinstance(response_data, dict) and 'result' in response_data:
                
                    top_list = response_data['result']['tops'][0]['keyword_data']['top']
                    
                    for top_item in top_list:
                        position = top_item.get('position')
                        domain = top_item.get('domain')
                        tops_info.append({'position': position, 'domain': domain})
                    
                    return pd.DataFrame(tops_info)
            else:
                print(f"Error in get_task_result: {response.status}")
                return None
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")

# async def get_domain_keywords(session, domain, se):
    
#     word_check = "what where who why trend analysis overview insight impact features comprehensive trending"
    
#     payload = {
#         "id": "1",
#         "method": "SerpstatDomainProcedure.getDomainKeywords",           
#         "params": {                                                       
#             "domain": domain,
#             "se": se,
#             "page": "1",
#             "size": "1000",
#             "sort": {
#                 "traff": "desc"
#             },
#             "keyword_contain_one_of_broad_match": str(word_check)
#         }
#     }
    
#     try:
#         async with session.post(url_main, json=payload) as response:
#             if response.status == 200:
#                 result = await response.json()
                
#                 if 'error' in result:
#                     error_code = result['error']['code']
#                     error_message = result['error']['message']
                    
#                     print(f"Error {error_code}: {error_message}")
#                     return None 
                
#                 if isinstance(result, dict) and 'result' in result:
#                     data_list = result['result']['data']
#                     extracted_data = [(item['url'], item['found_results']) for item in data_list]
#                     extracted_data = pd.DataFrame(extracted_data, columns=['url', 'found_results'])
#                     extracted_data = extracted_data.sort_values(by='found_results', ascending=False)
#                     print("extracted_data = ", extracted_data)
#                     return extracted_data
#             else:
#                 print(f"Error in get_domain_keywords: {response.status}")
#                 return None
            
#     except aiohttp.ClientResponseError as e:
#         print(f"Ошибка ответа клиента: {e}")
#     except aiohttp.ClientError as e:
#         print(f"Ошибка клиента aiohttp: {e}")
#     except asyncio.TimeoutError:
#         print("Тайм-аут запроса.")
#     except Exception as e:
#         print(f"Неизвестная ошибка: {e}")

# Определние статей на сайте
async def get_domain_keywords(message: Message, session: aiohttp.ClientSession, domain: str, se: str) -> pd.DataFrame:
    
    intentinformational_keywords = "how what why where when who guide tutorial best tips examples steps ideas ways strategy strategies compare comparison benefits advantages disadvantages review reviews how to learn explain overview introduction example explained meaning definition instructions help resources facts info information analysis summary description guide to methods techniques list listing process checklist explore discover history facts about ways to options alternatives types tutorials case study case studies pros and cons trends patterns background data statistics updates insights news reports report white paper ebook blog post article research study findings explanation reasons causes ways to methods of overview of types of effects of impact of difference between cost of cost comparison meaning of definition of review of reviews of how do how does how can why does why do why is who is who are what is what are where can when should which is which are can i can you could i could you should i should you"

    page = 1
    all_data = []

    # while True:
    while page < 5:
        # Формируем данные для запроса
        data = {
            "id": "1",
            "method": "SerpstatDomainProcedure.getDomainKeywords",
            "params": {
                "domain": domain,
                "se": se,
                "page": str(page),
                "size": "100",
                "sort": {
                    "traff": "desc"
                },
                "keyword_contain_one_of_broad_match": str(intentinformational_keywords)
            }
        }

        try:
            # Асинхронно отправляем запрос
            async with session.post(url_main, json=data) as response:
                if response.status == 200:
                    result = await response.json()

                    # Проверка на ошибки в ответе
                    if 'error' in result:
                        error_code = result['error']['code']
                        if error_code == 32017:  # Data not found
                            print("Data not found - завершение цикла.")
                            break
                        elif error_code == 32018:  # To get more than 60000 results use export report methods
                            print("Достигнуто ограничение на количество данных - завершение цикла.")
                            break
                        else:
                            print("Неизвестная ошибка:", result['error']['message'])
                            break

                    # Проверка на наличие данных
                    if isinstance(result, dict) and 'result' in result:
                        response_data = result['result']['data']
                        if response_data:
                            all_data.extend(response_data)  # Добавление данных в общий список
                            page += 1  # Увеличение номера страницы
                        else:
                            print("Нет данных на странице", page)
                            break

                else:
                    print(f"Ошибка при запросе: {response.status}")
                    break

        except aiohttp.ClientResponseError as e:
            print(f"Ошибка ответа клиента: {e}")
            break
        except aiohttp.ClientError as e:
            print(f"Ошибка клиента aiohttp: {e}")
            break
        except asyncio.TimeoutError:
            print("Тайм-аут запроса.")
            await message.answer(f"Тайм-аут запроса. Повторите попытку.")
            break
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")
            break

    # Создание DataFrame из всех собранных данных
    if all_data:
        domains = []
        found_results = []
        position = []
        traff = []

        for item in all_data:
            domains.append(item['url'])
            found_results.append(item['found_results'])
            position.append(item['position'])
            traff.append(item['traff'])

        df = pd.DataFrame({
            'url': domains,
            'found_results': found_results,
            'position': position,
            'traff': traff
        })
        
        # Сортировка для определения самых трафиковых
        df = df.sort_values(by=['found_results', 'position', 'traff'], ascending=[False, False, False])

        return df
    else:
        print("Нет данных для создания DataFrame.")
        return pd.DataFrame() 
    
# Получение ссылок на аналагичные статьи
async def get_url_competitors(message: Message, session: aiohttp.ClientSession, url: str, se: str) -> pd.DataFrame:
    
    payload = {
        "id": "1",
        "method": "SerpstatUrlProcedure.getUrlCompetitors",
        "params": {
            "se": f"{se}",
            "url": f"{url}",
            "sort": {"cnt": "desc"}
        }
    }

    try:
        async with session.post(url_main, json=payload, timeout=10) as response:
            if response.status == 200:
                result = await response.json()
                print(result)
                
                if isinstance(result, dict) and 'result' in result:
                    data_list = result['result'].get('data')
                    extracted_data = pd.DataFrame([(item['url']) for item in data_list])
                    return extracted_data 
                else:
                    # Обработка страниц, для которых не найдены конкуренты
                    pass
                
               
            else:
                print(f"Error in add_task: {response.status}")
                return pd.DataFrame() 
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
    
    return pd.DataFrame() 
        
# Получение конкурентов домена
async def get_competitors(message: Message, session: aiohttp.ClientSession, initial_url: str, se: str) -> pd.DataFrame:
    
    parsed_url = urlparse(initial_url)
    domain = parsed_url.netloc
    
    payload = {
        "id": 1,
        "method": "SerpstatDomainProcedure.getOrganicCompetitorsPage",
        "params": {
        "domain": f"{domain}",
        "se": f"{se}",
        "sort": 
            {"relevance":"desc"}
        }
    }

    try:
        async with session.post(url_main, json=payload, timeout=10) as response:
            if response.status == 200:
                result = await response.json()
                
                if isinstance(result, dict) and 'result' in result:
                    response_data = result['result']['data']
                    
                    # Инициализация списков для хранения данных
                    domains = []
                    relevances = []
                    
                    # Извлечение данных из каждого объекта response_data
                    for item in response_data:
                        domains.append(item['domain'])
                        relevances.append(item['relevance'])
                    
                    # Создание DataFrame
                    df = pd.DataFrame({
                        'domain': domains,
                        'relevance': relevances
                    })
                    
                    df = df.dropna()
                    df = df.drop_duplicates()
                    
                    df_sorted = df.sort_values(by='relevance', ascending=False)
                    df_sorted.reset_index(drop=True, inplace=True)
                    
                    # Добавить логику фильтрации конкурентов
                    
                    relevance_cutoff = 5
                    
                    df_sorted = df_sorted[df_sorted['relevance'] >= relevance_cutoff]
                    df_sorted = df_sorted.drop(df_sorted.index[0])
                    
                    return df_sorted
                
            else:
                print(f"Error in add_task: {response.status}")
                return pd.DataFrame()
            
    except aiohttp.ClientResponseError as e:
        print(f"Ошибка ответа клиента: {e}")
    except aiohttp.ClientError as e:
        print(f"Ошибка клиента aiohttp: {e}")
    except asyncio.TimeoutError:
        print("Тайм-аут запроса.")
        await message.answer(f"Тайм-аут запроса. Повторите попытку.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        
    return pd.DataFrame() 
