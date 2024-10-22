from openai import AsyncOpenAI
import requests
import aiohttp
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from aiogram.types import Message
from langchain_community.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer, util
import logging
from aiohttp import ClientSession
import src.serpstat as serpstat

from src.config import settings_pr

client = AsyncOpenAI(
    api_key=settings_pr.openai_api_key
)

INPUT_COF_4O = 0.00000125
OUTPUT_COF_4O = 0.000005

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Асинхронная функция для вставки заголовка после семантически схожей фразы с обработкой ошибок
async def insert_heading_after_semantic_phrase(text: str, phrase: str, heading: str) -> str:
    try:
        # Проверка входных данных
        if not text or not phrase or not heading:
            raise ValueError("Ошибка: Текст, фраза или заголовок не могут быть пустыми.")
        
        # Разбиваем текст на предложения для поиска места вставки
        sentences = text.split('. ')
        if len(sentences) < 2:
            raise ValueError("Ошибка: Текст слишком короткий для обработки.")

        # Получаем эмбеддинги для фразы и предложений текста
        try:
            phrase_embedding = model.encode(phrase, convert_to_tensor=True)
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании эмбеддингов: {str(e)}")

        # Находим предложение, которое максимально семантически похоже на фразу
        try:
            cosine_scores = util.pytorch_cos_sim(phrase_embedding, sentence_embeddings)
            best_match_index = cosine_scores.argmax().item()  # Получаем индекс наиболее похожего предложения
        except Exception as e:
            raise RuntimeError(f"Ошибка при вычислении семантической близости: {str(e)}")

        # Разделяем текст на две части: до и после найденного предложения
        before_sentence = '. '.join(sentences[:best_match_index + 1])
        after_sentence = '. '.join(sentences[best_match_index + 1:])
        
        # Форматирование заголовка в виде HTML-тега
        heading_tag = f'<h2>{heading}</h2><br>'
        
        # Объединяем части текста с вставленным заголовком
        new_text = before_sentence + heading_tag + after_sentence

        return new_text

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return f"Ошибка: {ve}"

    except RuntimeError as re:
        logging.error(f"RuntimeError: {re}")
        return f"Ошибка: {re}"

    except Exception as e:
        logging.error(f"Неизвестная ошибка: {e}")
        return "Произошла неизвестная ошибка при вставке заголовка."


# Определение места для вставки заголовков в текст
async def define_locations_for_headings(input_text: pd.DataFrame, headings: list[str]) -> dict[str, str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "heading_insertion",
                "description": """
                
                    Анализ текста статьи для определения мест, где логично вставить заголовки для улучшения структуры и восприятия информации. 
                    
                    ### Требования ###:
                    - Места ДОЛЖНЫ быть расположены так, чтобы не нарушать логический поток текста.
                    - Места не должны быть непосредственно в конце текста.
                    - Добавляй места как они идут в изначальном списке. 
                
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "insert_locations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "insert_location": {
                                        "type": "string",
                                        "description": "ТОЧНОЕ ДОСЛОВНОЕ словосочетание из текста статьи, после которого необходимо вставить заголовок. Возвращай дословную фразу из исходного текста статьи."
                                    }
                                },
                                "required": ["insert_location"]
                            },
                            "description": "Список мест в тексте для вставки заголовков"
                        }
                    },
                    "required": ["insert_locations"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    results = {}
    
    list_of_headings = headings.strip().split('\n')

    # Перебор каждого заголовка из списка
    for heading in list_of_headings:
        prompt = f"""
        
            Проанализируй следующий текст и определи логическое место для вставки заголовка "{heading}".
            Заголовок должен быть вставлен так, чтобы сохранялась целостность и структура текста, а также улучшалась читабельность и восприятие материала.

            Место для заголовка должно быть логичным продолжением мысли предыдущего абзаца или подзаголовка, и не должно нарушать текущий поток текста.

            Формат вывода:
            Место для заголовка: <описание места в тексте для вставки заголовка>

            Текст статьи:
            {input_text}
            
        """
        
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты — аналитик текста, который определяет места для вставки заголовков..."},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
        )
        
        if completion.choices[0].finish_reason != 'tool_calls':
            results[heading] = "None"
            continue

        try:
            # Извлечение результата и добавление его в словарь с заголовком
            res = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["insert_locations"][0]["insert_location"]
            results[heading] = res
        except (KeyError, ValueError, TypeError) as e:
            logging.error(f"Ошибка при анализе результата: {e}")
            results[heading] = "Error"

    return results

async def generate_keywords_for_article_gpt(title: str) -> str:
    
    # Промт для генерации ключевых слов на основе названия статьи
    promt = """
        Generate 10 SEO-friendly keywords for an article titled: '{title}'. 
        
        Output the keywords in one line without numbering or bullet points, separated by a single space, without any line breaks.
        
    """.format(title=title)
    
    try:
        # Взаимодействие с OpenAI API для генерации ключевых слов
        # completion = client.chat.completions.create(
        #     # model="gpt-4o",  
        #     model="o1-preview",
        #     messages= [
        #         {
        #             "role": "user", 
        #             "content": promt
        #         }
        #     ]
        # )
        
        completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a professional SEO content writer."},
                {"role": "user", "content": promt}
            ]
        )
        
        # Возвращаем сгенерированные ключевые слова
        return str(completion.choices[0].message.content) 
    except Exception as e:
        print(f"Error during keyword generation: {e}")
        return ""
    
    
    
async def rewrite_keywords_title(title: str, keywords: str, total_tokens_input_4o: int, total_tokens_output_4o: int):
    
    # Промт для генерации ключевых слов на основе названия статьи
    prompt = f"""
        Ты являешься экспертом по SEO и оптимизации контента. Твоя задача — взять следующий заголовок статьи и список ключевых слов, проанализировать их, а затем переформулировать заголовок, органично добавив в него релевантные ключевые слова из списка для улучшения SEO.

        Заголовок статьи: '{title}'

        Ключевые слова: {keywords}

        Новый заголовок должен:
        - Включать 1-2 ключевых слова из списка, где это уместно (приоритетные ключевые слова следует использовать в первую очередь).
        - Сохранять естественный, увлекательный поток и соответствовать целевой аудитории (например, маркетологи, технические специалисты, широкая публика).
        - Избегать чрезмерного использования ключевых слов (keyword stuffing).
        - Оставаться понятным и легко воспринимаемым для читателей.
        - Заголовок должен быть ограничен 60 символами для оптимального отображения в поисковых системах.
        - Сохранять соответствующий стиль текста (например, профессиональный, разговорный, новостной).
        
         - !!! УБРАТЬ любое упоминание компании или организации, которая написала эту статью !!!

        Пожалуйста, верни ТОЛЬКО переформулированный заголовок в виде одного предложения, без ковычек, скобок и лишних слов.
    """
    
    try:
        # Взаимодействие с OpenAI API для генерации ключевых слов
        # completion = client.chat.completions.create(
        #     # model="gpt-4o",  
        #     model="o1-preview",
        #     messages= [
        #         {
        #             "role": "user", 
        #             "content": promt
        #         }
        #     ]
        # )
        
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional SEO content writer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        total_tokens_input_4o += completion.usage.prompt_tokens
        total_tokens_output_4o += completion.usage.completion_tokens
        
        # Возвращаем сгенерированные ключевые слова
        return str(completion.choices[0].message.content), total_tokens_input_4o, total_tokens_output_4o
    except Exception as e:
        print(f"Error during keyword generation: {e}")
        return "", total_tokens_input_4o, total_tokens_output_4o

# Асинхронная функция для фильтрации заголовков и добавления картинок
async def define_location_for_picture(input_text: pd.DataFrame) -> list[str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "img_input",
                "description": "Анализ текста статьи для определения мест, где логично вставить изображение-мем для улучшения восприятия информации. Места определяются по введению новых концептов, описанию функций или эмоциональных моментов. Места ДОЛЖНЫ быть расположены равномерно по всему тексту!",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mem_locations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "insert_location": {
                                        "type": "string",
                                        "description": "ТОЧНОЕ ДОСЛОВНОЕ словосочетание из текста статьи, после которого необходимо вставить картинку-мем. Возвращай дословную фразу из исходного текста статьи."
                                    }
                                },
                                "required": ["insert_location"]
                            },
                            "description": "Список мест в тексте для вставки заголовка"
                        }
                    },
                    "required": ["mem_locations"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    prompt = """
    
        Проанализируй следующий текст и определи ключевые моменты, где вводится новый концепт, 
        описываются важные функции, процессы или присутствует эмоциональная окраска текста.
        
        Определи, после каких заголовков или в каких моментах текста логичнее всего добавить мем, чтобы усилить восприятие или сделать текст более живым и понятным.
        
        На основе контекста текста создай описание для мема:
        - Если введен новый концепт, мем должен помогать визуализировать идею в юмористической форме.
        - Если текст описывает процесс или сложную функцию, мем может иллюстрировать преувеличенные последствия использования этой функции.
        - Если текст имеет эмоциональную окраску (например, трудности или радость), мем должен усиливать этот эмоциональный отклик с помощью юмора.
        
        Пример вывода:
        
        Место для мема: После подзаголовка "Основные возможности Gemini: модели Ultra, Pro, Flash и Nano с расширенным окном контекста".
        Описание картинки: Мем, на котором изображены четыре персонажа, символизирующие разные модели Gemini. Например, сильный супергерой для Ultra, быстрый персонаж для Flash, сбалансированный герой для Pro и миниатюрный персонаж для Nano. Подписи под персонажами усиливают различия между моделями, добавляя юмористический контекст.
        
        Формат вывода:
        Место для мема: <описание, после какого места в тексте вставить мем>
        Описание картинки: <текстовое описание мема, соответствующего контексту статьи>
        
        Текст статьи:{input_text}
        
    """.format(input_text=input_text)
    
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты — аналитик текста, который находит места для вставки мемов..."},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
    )
    
    if completion.choices[0].finish_reason != 'tool_calls':
        return "None"

    try:
        res = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["mem_locations"]
        return res
    except (KeyError, ValueError, TypeError) as e:
        logging.error(f"Ошибка при анализе результата: {e}")
        return pd.DataFrame()
    
async def generate_keywords_for_article(url: str, title: str, session: aiohttp.ClientSession) -> str:
    
    try:
        
        se = "g_by"
        keywords = await serpstat.get_page_keywords(session, url, se)
        
        if not keywords:
            
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            keywords = await serpstat.get_page_keywords(domain, url, se)
            
            if not keywords:
                keywords = await generate_keywords_for_article_gpt(title)
                return keywords
            
            return keywords
        
        return keywords
        
    except Exception as e:
        print(f"Error during keyword generation: {e}")
        return ""

###############################################################################################################################################

# Обработчик ошибок
def log_error(message: str, error: Exception):
    logging.error(f"{message}: {error}")

async def gen_keyword_article(message: Message, url: str, article_title: str, session: ClientSession) -> str:
    
    total_tokens_input_4o, total_tokens_output_4o = 0, 0
    
    try:
        # Парсинг контента статьи
        loader = UnstructuredURLLoader(urls=[url])
        content = loader.load()

        # Проверяем, что контент не пустой
        if content and content[0].page_content:
            article_content = content[0].page_content
        else:
            raise ValueError("Content_error: Статья пуста или не была загружена")

    except Exception as e:
        log_error("Ошибка при парсинге контента статьи", e)
        return "Не удалось загрузить контент статьи."
    
    # try:
    # # Синхронный GET-запрос
    #     response = requests.get(url, timeout=10)
    #     response.raise_for_status()  # Проверяем, что запрос прошел успешно

    #     # Попытка установить правильную кодировку с заголовков ответа
    #     response.encoding = response.apparent_encoding

    #     # Получаем содержимое страницы
    #     text = response.text
    #     # Парсим содержимое страницы с помощью BeautifulSoup
    #     article_content = BeautifulSoup(text, 'html.parser')
        
    #     # Извлекаем заголовок статьи
    #     # article_title = soup.find('title').get_text() if soup.find('title') else "No Title Found"
    #     # print(f"Title of the article: {article_title}")

    # except requests.RequestException as e:
    #     print(f"Ошибка при запросе: {e}")

    try:
        # Промт удаления рекламы
        plan_query = f"""
        
            ### Роль ###:
                - Действуй как профессиональный аналитик текста.
                - Ты должен обработать предоставленный текст и выполнить следующие шаги, не изменяя исходный текст, за исключением удаления предложений, связанных с названием компании, её рекламой и заголовков.
                
            ### Требования ###:
                - В итоговый вывод ДОБАВЬ ТОЛЬКО изначальный текст, отфильтрованный от предложений, которые относятся к названию компании, её рекламы и заголовков.
                - НЕ ВЫВОДИ свои рассуждения, а ВЫВЕДИ ТОЛЬКО итоговый отфильтрованный текст.
                Сохраняй полностью структуру и содержание изначального текста за исключением удаленных частей.
                
            ### Инструкция ###:
                - Определи название компании: 
                    Используй только текст URL, не переходя по ссылке. Определи название компании по основному домену и ключевым словам в URL. Игнорируй поддомены и любые параметры после основного URL. Если в URL не содержится явно название компании, постарайся извлечь его по тематике или ключевым словам.
                    
                    Пример: для URL www.example.com/shop-offers названием компании будет "Example".
                    
                - Найди упоминания компании и её рекламу в тексте:

                    После того как определишь название компании, найди все упоминания этого названия в тексте.
                    Также найди предложения, связанные с рекламой этой компании (например, упоминания скидок, акций, специальных предложений).
                    
                    Пример рекламных предложений: "Скидка 50% на все товары компании Example", "Закажите сейчас и получите бесплатную доставку от Example".
                
                - Найди заголовки:
                    Определи и удали заголовки из текста (обычно заголовки выделены как отдельные строки (пустая строка с каждой стороны), начинающиеся с заглавной буквы или номера и небольшой длинны).
                    Заголовки могут быть основными (H1, H2, H3) или промежуточными разделами.
                
                - Фильтрация текста:
                    Удали все предложения, где есть упоминания названия компании и реклама.
                    Удали все заголовки, чтобы сохранить только текстовую часть статьи.
                    Оставь все остальные части текста без изменений.
                
                - Верни итоговый текст:
                    Верни исходный текст, из которого удалены все предложения, связанные с названием компании, рекламой и заголовки. Не добавляй собственных рассуждений или выводов.
                
            ### Данные на вход ###:
                - Текст на вход: {article_content}
                - URL страницы: {url}

        """
        
        response = await client.chat.completions.create(
        #     model="o1-preview",
        #     messages=[{"role": "user", "content": plan_query}]
        # )
        
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a professional SEO content writer."},
                {"role": "user", "content": plan_query}
            ]
        )
        
        total_tokens_input_4o += response.usage.prompt_tokens
        total_tokens_output_4o += response.usage.completion_tokens

        article_content = str(response.choices[0].message.content)
        print(article_content)
        article_content_len = len(article_content)

    except Exception as e:
        log_error("Ошибка при генерации плана статьи с помощью OpenAI", e)
        return "Не удалось сгенерировать план статьи."
    
    try:
        # Промт для выделения плана
        plan_query = f"""
            <tag>
                **Роль**: Ты выступаешь в роли **опытного аналитика и редактора**, который умеет работать с текстами разного формата. Твоя задача — проанализировать текст статьи и на его основе составить структурированный план. План должен отражать основную логику и ключевые разделы текста, сохраняя последовательность идей и структуры статьи.
            </tag>

            <tag>
                **Задача**: Получив текст статьи на вход, тебе необходимо составить план статьи, придерживаясь следующих рекомендаций:

                !!! СОБЛЮДАЙ ВСЕ ПРОПИСАННЫЕ ТРЕБОВАНИЯ В СЛЕДУЮЩЕМ РАЗДЕЛЕ !!!
            </tag>

            <tag>
                **Требования**:

                1. **Структурированность**: План должен быть логически выстроенным и отражать основные разделы статьи. Обозначь каждый пункт плана кратким описанием идеи или основного тезиса, заложенного в статье.
                
                2. **Иерархия**: План должен иметь ясную иерархию. Главные разделы (основные пункты плана статьи) статьи следует ОБЯЗАТЕЛЬНО обозначить в следущем формате: **Название раздела (пункта)**
                - НЕ ИСПОЛЬЗУЙ ПОДЗАГОЛОВИК, ПОДПУНКТЫ. ПЛАН ДОЛЖЕН СОСТОЯТЬ ТОЛЬКО ИЗ ОСНОВНЫХ РАЗДЕЛОВ. 
                - Не добавляй в заголовки шаблонные фразы, например введение, заключение, вывод, начало и другие
                - План ДОЛЖЕН включать в себя НЕ более 5 разделов (пунктов)
                
                3. **Последовательность**: Структура плана должна следовать за логикой и структурой оригинальной статьи. Учитывай введение, основную часть и заключение, даже если они не выделены явно в тексте.

                4. **Лаконичность**: Старайся не перегружать план деталями. Кратко излагай основную мысль каждого раздела статьи, сохраняя общий смысл текста.
                
                5. **Вывод информации**: 
                    - Выводи ОБЯЗАТЕЛЬНО исключительно план статьи, без других элементов, например фраз "план статьи" и других
                    - Выводи план на русском.
                    - Не используй между пунктами плана пробел.
                    - Каждый пункт плана выделяй с двух сторон символом (переходом на следующую строку и ** в начале и в конце) \n**
                    - Каждый новыц пункт плана, начинать с новой строки!
                    - Между ОБЯЗАТЕЛЬНО пунктами плана не добавляй enter
            </tag>

            <tag>
                **Текст статьи на вход**: ["{article_content}"]
            </tag>
        """
        
        response = await client.chat.completions.create(
        #     model="o1-preview",
        #     messages=[{"role": "user", "content": plan_query}]
        # )
        
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional SEO content writer."},
                {"role": "user", "content": plan_query}
            ]
        )
        
        total_tokens_input_4o += response.usage.prompt_tokens
        total_tokens_output_4o += response.usage.completion_tokens
        
        article_plan = response.choices[0].message.content
        print(f"План статьи: {article_plan}")

    except Exception as e:
        log_error("Ошибка при генерации плана статьи с помощью OpenAI", e)
        return "Не удалось сгенерировать план статьи."
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_text_meanings",
                "description": "Analyze the input text, count distinct, non-overlapping meanings, and extract key sentences without losing any part of the text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "meanings": {
                            "type": "array",
                            "description": "List of distinct (minimum 5 in the all text), non-overlapping meanings extracted from the text, each with ONLY a start and end sentence. Ensure NO part of the text is lost between meanings.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_sentence": {
                                        "type": "string",
                                        "description": "ONLY first sentence of the meaning."
                                    },
                                    "end_sentence": {
                                        "type": "string",
                                        "description": "ONLY last sentence of the meaning."
                                    },
                                    "block_text": {
                                        "type": "string",
                                        "description": "The full text of the meaning block, ensuring no part of the text is lost between meanings."
                                    }
                                },
                                "required": ["start_sentence", "end_sentence", "block_text"],
                            }
                        },
                        "count": {
                            "type": "integer",
                            "description": "Total number of distinct, non-overlapping meanings found."
                        }
                    },
                    "required": ["meanings", "count"],
                    "additionalProperties": False,
                }
            }
        }
    ]
    
    prompt = (
        f"Analyze the following text:\n\n{article_content}\n\n"
        "1. Identify distinct meanings in the text. Ensure each meaning is clearly separated, non-overlapping, and that no part of the text is lost.\n"
        "2. For each meaning, return its first and last sentences, and the entire block of text that constitutes that meaning. Preserve the continuity of the text between meanings.\n"
        "3. Provide the total number of meanings and a list of objects, where each object contains the first and last sentences of the meaning, and the full text of the block."
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a link analyzer"},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
    )
    
    total_tokens_input_4o += response.usage.prompt_tokens
    total_tokens_output_4o += response.usage.completion_tokens

    if response.choices[0].finish_reason != 'tool_calls':
        return ""

    try:
        initial_article = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["meanings"]
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка при анализе результата: {e}")    
    
    
    block_texts = []

    # Loop through each object and extract block_text
    for meaning in initial_article:
        block_texts.append(meaning["block_text"])
        
    # print(block_texts)
    
    rewrite_article_content = str(" ")
    
    for section in block_texts:
        # Здесь можно выполнить любое действие для каждого элемента
        
        article_query = f"""
        
            <tag>
                **Роль**: Ты выступаешь в роли **опытного редактора и автора**, который умеет перерабатывать статьи, сохраняя их структуру и ключевые идеи. Твоя задача — переписать фрагмент текста соответствуя правилам и улучшая исходный материал.
            </tag>

            <tag>
                **Задача**: На основе исходного фрагмента текста, тебе необходимо переписать изначальный текст, строго придерживаясь следующих требований:

                !!! СТРОГО СОБЛЮДАЙ ВСЕ ПРОПИСАННЫЕ ТРЕБОВАНИЯ В СЛЕДУЮЩЕМ РАЗДЕЛЕ !!!
            </tag>

            <tag>
                **Требования**:
            
                    ### 1. Следование плану:
                    - Каждый акпект изначального текста должен быть максимально подробно и полно (ОБЯЗАТЕЛЬНО добавляй ИЗБЫТОЧНОЕ количество полезного контента) раскрыт в отдельном разделе текста.
                    - Длинна итогового текста ДОЛЖНА СОСТАВЛЯТЬ {len(section)} символов. ДАННОН ТРЕБОВАНИЕ !!!ОБЯЗАТЕЛЬНОЕ!!!

                    ### 2. Оригинальность:
                    - Переписывание своими словами: Используй исходный текст как основу для анализа, но пиши текст от себя, как автор. Это поможет избежать однообразия и придаст тексту индивидуальность.
                    Избегание копий: Никогда не копируй фразы или предложения из исходного текста. Всегда старайся передавать идеи по-своему, так, чтобы новый текст был полностью уникален.
                    - Эмоциональные акценты и примеры: Добавляй живые примеры и метафоры, которые отражают твою личную точку зрения и глубину анализа темы. Вместо того чтобы просто пересказывать, постарайся показать своё видение, делая статью более запоминающейся и интересной.
                    Избегание клише: Уходи от формальных клише и шаблонных выражений, которые часто встречаются в нейтральных текстах. Введи разнообразие в язык и стиль, чтобы читатель чувствовал увлечённость и энергию в тексте.
                    - Разнообразие подачи: Используй разные синтаксические конструкции, чередуй короткие и длинные предложения, чтобы текст был динамичным и удерживал внимание. Это избавит от однообразия и перегруженности.
                    
                    ### 3. Углубление материала:
                    - Добавление анализа и примеров: Каждая часть статьи должна быть тщательно проработана. Вместо поверхностного рассмотрения темы углубляйся в детали, давая чёткие и актуальные примеры, которые помогут читателю лучше понять материал.
                    - Аналитические выводы: Если исходный текст недостаточно глубок, дополняй его собственными выводами и анализом. Включай примеры из реальной жизни или отраслевые кейсы, которые обогащают текст и делают его полезным для читателя.
                    - Баланс деталей: Важно избегать перегрузки текста избыточной информацией. Стремись к тому, чтобы каждая деталь усиливала основную мысль, а не отвлекала от неё.
                    
                    ### 4. Стиль и язык:
                    - Профессионализм и доступность: Пиши на русском языке в профессиональном, но доступном стиле, ориентируясь на широкую аудиторию. Убедись, что текст легко воспринимается как специалистами, так и теми, кто только знакомится с темой.
                    - Ясность и простота: Избегай сложных и громоздких предложений. Чётко и ясно излагай мысли, разбивая длинные абзацы на более компактные, чтобы улучшить читаемость.
                    - Акцент на уникальность: Используй собственный голос, чтобы текст был интересным, избегай механистичности и сухого изложения. Это придаст тексту индивидуальность и повысит его ценность для аудитории.

                    ### 5. Логичность и структура:
                    - Следи за плавным переходом между разделами, обеспечивая логическую связь между ними.
                    - НЕ ИСПОЛЬЗУЙ слово "заключение, таким образом, в итоге, наконец, и так " в тексте нигде! и другие шаблонные слова для начала заключительной части.
                    - НЕ выводи название стать, автора, ссылки на другие статьи, описание темы, содержание, комментарии
                    - УБРАТЬ из текста упоминание компании, которая написала данную статью, любую саморекламу
                    - ОБЯЗАТЕЛЬНО УБРАТЬ со всего текста любую рекламу
                    - !!! НЕ ДОБАВЛЯЙ в текст никакие абзатцы !!!
                    
                    ### 6. Дополнительные требования:
                    -  СПИСКИ: При добавлении любых списков, формат списков ДОЛЖЕН быть следующий: - (нет вводного слова или фразы, а сразу контент пункта) Контекнт пункта списка. Следуй данному пункту ОБЯЗАТЕЛЬНО!!!
            </tag>

            <tag>
                **Текст на вход для рерайта**: ["{section}"]
            </tag>
        ""
            
        """
        
        response = await client.chat.completions.create(
        #     model="o1-preview",
        #     messages=[{"role": "user", "content": plan_query}]
        # )
        
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are a professional SEO content writer."},
                    {"role": "user", "content": article_query}
                ]
        )
        
        total_tokens_input_4o += response.usage.prompt_tokens
        total_tokens_output_4o += response.usage.completion_tokens
        
        article = str(response.choices[0].message.content)
        rewrite_article_content += article
        
        print(f"Количество символов в изначальном тексте: {article_content_len} символов.\nКоличество символов в итоговом тексте: {len(rewrite_article_content)} символов.\nПотери в %: {((len(rewrite_article_content) - article_content_len) / article_content_len) * 100}")
    
    print(rewrite_article_content)
    
    ###########################################################################################################################
    # Добавление ключевых слов
    ###########################################################################################################################

    # Предполагается, что `generate_keywords_for_article` — это функция для получения ключевых слов через API (например, Serpstat)
    try:
        # Список ключевых слов, который генерируется по заголовку статьи
        keywords = await generate_keywords_for_article(url, article_title, session)

        # Генерация финальной статьи с учетом ключевых слов
        keyword_insertion_query = f"""
        
            ### Роль ###: 
                - Ты — опытный SEO-специалист. Твоя задача — добавить в текст ключевые слова, строго придерживаясь указанных правил.

            ### Задача ###:
                - Твоя основная задача — встроить ключевые слова в исходный текст статьи, не изменяя его объем и структуру. 
                
                !!! СТРОГО ЗАПРЕЩАЕТСЯ удалять или сокращать оригинальный текст !!! 
                
                Ключевые слова должны быть добавлены естественно, без нарушения целостности текста и без их выделения. Текст должен полностью сохранить свою исходную длину.

            ### Требования ###:
                ## Интеграция ключевых слов ##:
                    - Поддерживай плотность ключевых слов на уровне 1-2%, но не нарушай объем текста.
                    - Никакие ключевые слова не должны выделяться или отличаться от основного текста. Они должны быть органично вписаны в текст.
                    - Запрещается изменять или удалять исходные элементы текста.
                    Объем текста:

                    - Объем текста должен остаться точно таким же — {len(rewrite_article_content)} символов.
                    - Ты можешь добавлять примеры или уточнения, чтобы естественно вписать ключевые слова, но текст не должен быть короче исходного объема.
                    
                    # Язык и стиль #:
                        - Весь текст должен оставаться на русском языке.
                        - Пиши текст в том же стиле, который использован в исходной статье, чтобы сохранить его целостность.
                
            ### Входные данные ###:
                - Ключевые слова: {keywords}
                - Текст статьи: {rewrite_article_content}
            
        """

        response = await client.chat.completions.create(
        #     model="o1-preview",
        #     messages=[{"role": "user", "content": keyword_insertion_query}]
        # )
        
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional SEO content writer."},
                {"role": "user", "content": keyword_insertion_query}
            ],
            # temperature=0.8
        )
        
        total_tokens_input_4o += response.usage.prompt_tokens
        total_tokens_output_4o += response.usage.completion_tokens

        key_words_article = response.choices[0].message.content
        print(f"Статья с ключевыми словами = {key_words_article}")
            
    except Exception as e:
        log_error("Ошибка при получении ключевых слов", e)
        return "Не удалось получить ключевые слова для статьи."
    
    try:
        # Добавление абзацев
        # result = await define_locations_for_headings(input_text=key_words_article, headings=article_plan)
        
        # if not result:
        #     raise ValueError("Не удалось найти места для вставки заголовков.")
        
        # print(result)
        
        # list_of_headings = article_plan.strip().split('\n')
        
        # # Выводим результат для проверки
        # for heading, location in result.items():
        #     if location:
        #         key_words_article = await insert_heading_after_semantic_phrase(text=key_words_article, phrase=location, heading=heading)
        #         print(f"Заголовок: {heading}, Место вставки: {location}")
        #     else:
        #         print(f"Не удалось найти место для заголовка: {heading}")
        
        keywords_article_title, total_tokens_input_4o, total_tokens_output_4o = await rewrite_keywords_title(article_title, keywords, total_tokens_input_4o, total_tokens_output_4o)

        print(f"total_tokens = {total_tokens}")
        print(f"Количество символов в изначальном тексте: {article_content_len} символов.\n"
            f"Количество символов в итоговом тексте: {len(key_words_article)} символов.\n"
            f"Потери в %: {((len(key_words_article) - article_content_len) / article_content_len) * 100}")
        print(f"Ключевые слова: {keywords}")

        price = (total_tokens_input_4o * INPUT_COF_4O) + (total_tokens_output_4o * OUTPUT_COF_4O)
        await message.answer(f"Цена статьи = {price}$\nКлючевые слова: {keywords}")

    except ValueError as ve:
        log_error("Ошибка при обработке заголовков", ve)
        return "Ошибка: некорректные данные для вставки заголовков."
    except Exception as e:
        log_error("Ошибка при получении ключевых слов", e)
        return "Не удалось получить ключевые слова для статьи."

    return key_words_article, keywords_article_title

