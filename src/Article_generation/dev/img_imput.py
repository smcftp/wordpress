import asyncio
import aiofiles
import logging
from openai import AsyncOpenAI

import aiohttp
import re
import webbrowser
import json
import pandas as pd
import time
from xhtml2pdf import pisa
from sentence_transformers import SentenceTransformer, util
import os
import base64
import uuid

from src.config import settings_pr

client = AsyncOpenAI(
    api_key=settings_pr.openai_api_key
)

# Инициализация модели
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Асинхронная функция для вставки изображений
async def insert_image_after_semantic_phrase(text: str, phrase: str, image_path: str, article_title: str) -> str:
    # Разбиваем текст на предложения для поиска места вставки
    sentences = text.split('. ')
    
    # Получаем эмбеддинги для фразы и предложений текста
    phrase_embedding = model.encode(phrase, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Находим предложение, которое максимально семантически похоже на фразу
    cosine_scores = util.pytorch_cos_sim(phrase_embedding, sentence_embeddings)
    best_match_index = cosine_scores.argmax().item()  # Получаем индекс наиболее похожего предложения

    # Разделяем текст на две части: до и после найденного предложения
    before_sentence = '. '.join(sentences[:best_match_index + 1])
    after_sentence = '. '.join(sentences[best_match_index + 1:])
    
    response = requests.get(image_path)
    if response.status_code == 200:
        # Преобразуем изображение в base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
    
        image_height = 400
        image_width = 400
        
        # Вставляем изображение с указанием размера
        # image_tag = f'<div class="image-container"><img src="{os.path.abspath(image_path)}" alt="Inserted Image" width="{image_width}" height="{image_height}"></div><br>'
        
        image_tag = f'<div class="image-container"><img src="data:image/png;base64,{image_base64}" alt="Inserted Image" width="{image_width}" height="{image_height}"></div><br>'

        # Объединяем части текста с вставленным изображением
        new_text = before_sentence + image_tag + after_sentence
        
        return new_text
    
    else:
        return text

# Асинхронная функция для загрузки текстового файла
async def load_txt_file(file_path):
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()
    except FileNotFoundError:
        return "Файл не найден."
    except Exception as e:
        return f"Произошла ошибка: {e}"

async def process_text(text: str) -> str:
    html_content = ""
    lines = text.split('\n')
    inside_list = False

    # Regular expression to match all pairs of ** for bold text
    bold_pattern = re.compile(r"\*\*(.+?)\*\*")

    for line in lines:
        line = line.strip()  # Remove extra spaces

        if line.startswith('-'):
            # Open a new list if not already inside a list
            if not inside_list:
                html_content += "<ul>\n"
                inside_list = True
            
            # Remove the '-' and format the list item
            line = line[1:].strip()

            # Replace **...** with <b>...</b> globally using regex
            line = bold_pattern.sub(r"<b>\1</b>", line)
            html_content += f"<li>{line}</li>\n"
        else:
            # Close the list if it was opened
            if inside_list:
                html_content += "</ul>\n"
                inside_list = False

            # Replace **...** with <b>...</b> globally using regex
            line = bold_pattern.sub(r"<b>\1</b>", line)
            if line:  # Ensure line is not empty
                html_content += f"<p>{line}</p>\n"

    # Close the list if it's still open
    if inside_list:
        html_content += "</ul>\n"

    return html_content

# Асинхронная функция для преобразования текста в HTML
async def text_to_html(text: str, article_title: str, file_name: str) -> str:
    print("Начало!")
  
    processed_text = await process_text(text)
    
    # Начало HTML-документа с вашим шаблоном
    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text with Image</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f0f0f5;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 800px;
                    margin: 20px;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    font-size: 1.1rem;
                    line-height: 1.6;
                }}
                h1 {{
                    font-size: 2rem;
                    color: #222;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .image-container {{
                    display: flex;
                    justify-content: center;
                    margin: 20px 0;
                }}
                img {{
                    border-radius: 12px;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    max-width: 100%;
                    height: auto;
                }}
                p {{
                    text-align: justify;
                    margin-bottom: 20px;
                }}

                /* Адаптация под экраны планшетов */
                @media (max-width: 768px) {{
                    .container {{
                        max-width: 90%;
                        padding: 15px;
                        font-size: 1rem;
                    }}
                    h1 {{
                        font-size: 1.8rem;
                    }}
                    p {{
                        font-size: 1rem;
                    }}
                }}
                @media (max-width: 480px) {{
                    .container {{
                        max-width: 100%;
                        margin: 10px;
                        padding: 10px;
                        font-size: 0.9rem;
                    }}
                    h1 {{
                        font-size: 1.5rem;
                    }}
                    p {{
                        font-size: 0.9rem;
                    }}
                    .image-container {{
                        margin: 15px 0;
                    }}
                    img {{
                        width: 100%;
                        border-radius: 8px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{article_title}</h1>
                {processed_text}
            </div>
        </body>
      </html>
    """

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    
    async with aiofiles.open(file_name, 'w', encoding='utf-8') as file:
        await file.write(html_content)
    
    return html_content

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
                                    },
                                    # "image_description": {
                                    #     "type": "string",
                                    #     "description": """
                                        
                                    #         !!! Description must be in English !!!
                                            
                                    #         The image should be a highly creative and humorous MEME that fits perfectly with the context of the surrounding text. If a new concept is introduced in the text, the meme should help visualize the idea in a light-hearted, funny way, making the concept easier to grasp. For instance, if the text describes a particular object, such as a new AI model, the meme should feature the name of the object prominently in the image (e.g., the name of the model written on a banner, sign, or any creative object in the meme).

                                    #         If the text describes a process or a complicated function, the meme should exaggerate the consequences of using the function humorously (e.g., a character struggling to handle something complex, only for AI to step in and solve it effortlessly in seconds). The exaggerated reaction could be depicted through exaggerated facial expressions or ridiculous physical actions, such as someone sweating profusely while trying to complete the task manually.

                                    #         In cases where the text has an emotional tone, whether it’s about frustration, excitement, or triumph, the meme should amplify that emotion using humor. For example, if the text describes joy, the meme could show an ecstatic character celebrating in an over-the-top way, such as throwing confetti, dancing wildly, or even being lifted into the air by balloons. If frustration is described, the meme could show a character dramatically pulling their hair out, breaking objects in frustration, or staring at a computer screen with eyes wide open in disbelief.

                                    #         Overall, the memes should be visually engaging, quirky, and unexpected to keep the reader entertained and make the concepts more memorable. Incorporate humorous details, like labels, thought bubbles, or exaggerated features, that tie directly into the subject matter or keywords mentioned in the text. The goal is to surprise and amuse the audience while making the content of the article easier to understand and enjoy.
                                    #     """    
                                    # }
                                    # "image_description": {
                                    #     "type": "string",
                                    #     "description": """
                                        
                                    #         !!! Description must be in English !!!

                                    #         The image should be a thoughtful and visually informative illustration that fits perfectly with the context of the surrounding text. If a new concept is introduced in the text, the image should help visualize the idea in a clear and professional manner, enhancing the reader's understanding. For example, if the text describes a particular object, such as a new AI model, the image should feature the object or its components in a visually appealing way, possibly showing diagrams, key features, or labels that clarify its function (e.g., a flowchart of the model’s architecture or a graphical representation of its capabilities).

                                    #         If the text describes a process or a complicated function, the image should clearly depict the stages of the process or the interaction between components, highlighting the main steps in a structured and easy-to-follow format. For instance, if the text discusses data processing or a machine learning pipeline, the image might show a step-by-step flow of how the data moves through the system, with concise annotations explaining each stage.

                                    #         In cases where the text conveys an emotional tone—such as excitement, frustration, or triumph—the image should reflect the gravity of those emotions through a more subdued, professional style. For example, if the text describes a significant breakthrough, the image might showcase an achievement through a striking visual, such as a graph demonstrating the results or a celebratory visual that remains dignified and respectful, such as a person reaching a goal in a serene landscape.

                                    #         Overall, the images should be visually informative, professional, and aligned with the subject matter to reinforce the key concepts and messages of the text. Incorporate details like icons, labels, and subtle design elements that directly relate to the subject or keywords in the text. The goal is to educate and engage the audience, making the content of the article more accessible and easier to comprehend while maintaining a polished and professional appearance.

                                    #     """    
                                    # }
                                    "image_description": {
                                        "type": "string",
                                        "description": """
                                        
                                            !!! Description must be in English !!!

                                            The image should be a thoughtful and visually informative illustration that fits perfectly with the context of the surrounding text. If a new concept is introduced in the text, the image should help visualize the idea in a clear and professional manner, enhancing the reader's understanding. For example, if the text describes a particular object, such as a new AI model, the image should feature the object or its components in a visually appealing way, possibly showing diagrams, key features, or labels that clarify its function (e.g., a flowchart of the model’s architecture or a graphical representation of its capabilities).

                                            If the text describes a process or a complicated function, the image should clearly depict the stages of the process or the interaction between components, highlighting the main steps in a structured and easy-to-follow format. For instance, if the text discusses data processing or a machine learning pipeline, the image might show a step-by-step flow of how the data moves through the system, with concise annotations explaining each stage.

                                            In cases where the text conveys an emotional tone—such as excitement, frustration, or triumph—the image should reflect the gravity of those emotions through a more subdued, professional style. For example, if the text describes a significant breakthrough, the image might showcase an achievement through a striking visual, such as a graph demonstrating the results or a celebratory visual that remains dignified and respectful, such as a person reaching a goal in a serene landscape.

                                            ### Stylistic Parameters ###:
                                            
                                                - Color Palette: The images should primarily feature natural and muted tones, with an emphasis on warm, earthy colors like brown, beige, and golden to create a soft and calming atmosphere. In some cases, use cooler tones like blue or cyan to create contrast, especially when highlighting specific elements. Additionally, some images may incorporate brighter and pastel colors (e.g., images with abstract themes or objects like clocks) to emphasize the surreal or conceptual nature.

                                                - Textures and Patterns: Use smooth and subtle textures in realistic scenes to create a sense of softness and depth, with seamless color transitions (e.g., scenes involving reading or writing). For more abstract or surrealistic images, keep textures minimalistic and clean to emphasize conceptual clarity. Geometric patterns, such as mosaic designs, may be used to evoke historical or cultural aesthetics (e.g., Roman-inspired themes).

                                                - Composition: The composition should be balanced and well-structured, especially in realistic scenes, where symmetry or slight asymmetry is used to focus on the interaction between characters or objects (e.g., reading or counting coins). In surreal or conceptual images, the composition should be more dynamic and asymmetrical, with unusual shapes or forms (e.g., abstract clocks) taking a central role to emphasize key ideas. Always ensure the composition draws attention to crucial details.

                                                - Shape and Object Style: Objects in the images should range from highly detailed and realistic (e.g., human figures, everyday objects) to stylized and abstract (e.g., surreal clocks with distorted proportions). Realistic objects should have natural proportions and fine details, while abstract or conceptual elements should play with shape and form to give the image philosophical or conceptual depth.

                                                - Lighting and Shadows: Use soft, natural lighting to emphasize key details and create a peaceful atmosphere in realistic scenes (e.g., a person writing in a forest). In surreal images, lighting should play a conceptual role, such as illustrating the passage of time or space (e.g., a clock set against a bright sky). Shadows should be subtle and used primarily for accentuating key elements without overwhelming the composition.

                                                - Theme: The themes of the images should vary from everyday scenes (e.g., reading a book, counting coins) to philosophical or surreal reflections on time, life, and meaning (e.g., a clock with a missing segment). Themes should center around concepts like knowledge, life, work, time, and spirituality, with some images highlighting the connection between humans and nature, while others explore more abstract or everyday topics.

                                                - Text or Graphic Elements: Text should not be present in the images, except where books or written documents are naturally included as part of the scene, emphasizing the importance of knowledge and literature. Graphic elements such as mosaic patterns or abstract designs may be used to enhance the visual style and add historical or philosophical context to the images.

                                                - Overall Style: The dominant style of the images should be a blend of realism and surrealism. Realistic scenes should showcase high levels of detail and depict everyday life, while surreal images should experiment with space, time, and conceptual forms. The common stylistic features across all images should include the use of warm, natural tones, balanced compositions, and an emphasis on the interaction between objects and people. The mood should be calm, reflective, and meditative, with elements of philosophy and spirituality.

                                        """    
                                    }
                                },
                                "required": ["insert_location", "image_description"]
                            },
                            "description": "Список мест в тексте для вставки изображений с описаниями соответствующих мемов."
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


async def rephrase_image_query(image_query: str) -> str:
    """Rephrases the image query to avoid content policy violations with error handling."""
    
    prompt = f"""
    
        Your task is to rephrase image queries to ensure they comply with content policies and are suitable for generating images. You have extensive experience in avoiding content policy violations and ensuring that queries are neutral, appropriate, and effective.

        You must always follow these guidelines:
        - Ensure the rephrased query is neutral and suitable for generating an image.
        - Avoid any content that might violate OpenAI content policies.
        - Translate the query to English if it is in another language.
        - Maintain the essence of the original query while ensuring it is appropriate.

        Your rephrased query must be clear and free of any content that might be considered inappropriate or offensive. Ensure that the rephrased query aligns with all content policies.
        Query to rephrase: {image_query}
        
    """
    
    try:
        # Отправляем запрос к API OpenAI для переформулировки запроса на изображение
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты — аналитик текста, который находит места для вставки мемов..."},
                {"role": "user", "content": prompt}
            ],
        )
        # Извлекаем и возвращаем переформулированный запрос
        rephrased_query = str(completion.choices[0].message.content)
        return rephrased_query

    except Exception as e:
        # Ловим все другие ошибки
        logging.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please try again later."

# Асинхронная функция для добавления изображений в статью
async def add_img_to_textarticle(text_article: str, article_title: str, img_gen: bool) -> str:
    input_text = text_article
    
    file_name = f"article_{uuid.uuid4().hex}.png"  # Имя вашего файла
    
    if img_gen == True:
        
        max_retries = 5  
        retries = 0
        
        while retries < max_retries:
            try:
                places_to_insert_img = await define_location_for_picture(input_text)
                
                # Проверяем, что возвращен список
                if isinstance(places_to_insert_img, list):
                    break
                else:
                    raise ValueError("Ожидался список, но получен другой тип данных")
            
            except (KeyError, ValueError, TypeError) as e:
                retries += 1
                logging.error(f"Ошибка при анализе результата: {e}. Повтор {retries} из {max_retries}")
                
            except Exception as e:
                logging.error(f"Неизвестная ошибка: {e}")
                return ""

        # logging.error(f"Не удалось получить правильный результат после {max_retries} попыток")
        
        formatted_output = input_text
        for insertion in places_to_insert_img:
            word = insertion['insert_location']
            prompt = """
            
                Ты являешься креативным генератором мемов. Твоя задача — создать забавное изображение-мем, основанное на описании ниже. Мем должен быть юмористическим, но при этом понятным для широкой аудитории.

                Описание изображения:
                "{mem_description}"

                Создай изображение с данным описанием, чтобы оно соответствовало контексту и передавало юмор в визуальной форме.
                
            """.format(mem_description=insertion['image_description'])

            image_url = None
            max_attempts = 5
            attempt = 0
            
            while not image_url and attempt < max_attempts:
                try:
                    # Пытаемся сгенерировать изображение
                    response = await client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                    )
                    image_url = response.data[0].url
                    
                    # Проверка, что ссылка на изображение действительно получена
                    if image_url:
                        # Генерируем уникальное имя для файла и сохраняем его
                        pass
                    else:
                        logging.info(f"Пустая ссылка на изображение. Попытка {attempt + 1} из {max_attempts}. Повтор запроса...")
                        # Если не удалось получить изображение, переформулируем запрос
                        prompt = await rephrase_image_query(prompt)

                except Exception as e:
                    logging.error(f"Ошибка при генерации изображения: {e}. Попытка {attempt + 1} из {max_attempts}.")
                    
                    # Переформулируем запрос на изображение
                    prompt = await rephrase_image_query(prompt)
                
                attempt += 1

            formatted_output = await insert_image_after_semantic_phrase(formatted_output, word, image_url, article_title)
        
        output_html = await text_to_html(formatted_output, article_title, file_name)

        # # Открываем HTML файл в браузере
        # webbrowser.open('file://' + os.path.realpath(output_html))
        
        htmp_path = os.path.abspath(file_name)
        
        return htmp_path
    
    else: 
        
        formatted_output = await process_text(input_text)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text with Image</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f0f0f5;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 800px;
                    margin: 20px;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    font-size: 1.1rem;
                    line-height: 1.6;
                }}
                h1 {{
                    font-size: 2rem;
                    color: #222;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .image-container {{
                    display: flex;
                    justify-content: center;
                    margin: 20px 0;
                }}
                img {{
                    border-radius: 12px;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    max-width: 100%;
                    height: auto;
                }}
                p {{
                    text-align: justify;
                    margin-bottom: 20px;
                }}

                /* Адаптация под экраны планшетов */
                @media (max-width: 768px) {{
                    .container {{
                        max-width: 90%;
                        padding: 15px;
                        font-size: 1rem;
                    }}
                    h1 {{
                        font-size: 1.8rem;
                    }}
                    p {{
                        font-size: 1rem;
                    }}
                }}
                @media (max-width: 480px) {{
                    .container {{
                        max-width: 100%;
                        margin: 10px;
                        padding: 10px;
                        font-size: 0.9rem;
                    }}
                    h1 {{
                        font-size: 1.5rem;
                    }}
                    p {{
                        font-size: 0.9rem;
                    }}
                    .image-container {{
                        margin: 15px 0;
                    }}
                    img {{
                        width: 100%;
                        border-radius: 8px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{article_title}</h1>
                {formatted_output}
            </div>
        </body>
      </html>
    """

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    
    async with aiofiles.open(file_name, 'w', encoding='utf-8') as file:
        await file.write(html_content)
        
    return os.path.abspath(file_name)
    
    
