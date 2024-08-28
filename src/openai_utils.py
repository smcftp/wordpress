from openai import AsyncOpenAI
import json

import pandas as pd

from src.config import settings_pr

client = AsyncOpenAI(
    api_key=settings_pr.openai_api_key
)

async def analyze_keywords_with_openai(keyword_list):
    # код функции
    pass


async def check_article_exists(link: str) -> bool:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_is_url_article",
                "description": "Analyze the link and understand whether it was entered on the page with the article or not.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_article": {
                            "type": "boolean",
                            "description": "True if the link leads to an article, False otherwise",
                        }
                    },
                    "required": ["is_article"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    prompt = (
        f"Analyze the provided URL: {link}. Based on its structure, content, and common patterns "
        "found in article URLs, determine if it leads to an article page or another type of page. "
        "Return 'True' if it is an article, otherwise return 'False'."
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a link analyzer"},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
    )

    if completion.choices[0].finish_reason != 'tool_calls':
        return False

    try:
        bool_res = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["is_article"]
        return bool_res
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка при анализе результата: {e}")
        return False
      
async def filter_article_titles(titles: pd.DataFrame) -> list[str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "filter_titles",
                "description": "If you receive a list or pandas dataframe of article titles. Analyze the provided list (or pandas dataframe) of titles to determine if each title is likely to be an article title. Filter out titles that are unlikely to be articles, it is mean if the entry is not similar to the title of the article, it looks like an incorrect value, then DELETE it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "valid_titles": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of titles that are filtered from incorrect values."
                        }
                    },
                    "required": ["valid_titles"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    titles = titles['title'].tolist()

    prompt = (
        "Analyze the provided list or pandas dataframe of titles. Determine which titles are likely to be the names of articles based on common patterns and characteristics of article titles. "
        "Return a list of titles that are most likely to be articles, and filter out those that are not."
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a title analyzer."},
            {"role": "user", "content": prompt + "\n\n" + "\n".join(titles)}
        ],
        tools=tools,
    )
    
    print(completion.choices[0].finish_reason)
    
    if completion.choices[0].finish_reason != 'tool_calls':
        return titles

    try:
        res = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["valid_titles"]
        return res
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка при анализе результата: {e}")
        return False


async def filter_article_titles_theme(titles: pd.DataFrame, theme: str) -> list[str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "filter_titles",
                "description": "If you receive a list or pandas dataframe of article titles. Analyze the provided list (or pandas dataframe) of titles to determine if each title is likely to be an article title. Filter out titles that are unlikely to be articles, it is mean if the entry is not similar to the title of the article, it looks like an incorrect value, then DELETE it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "valid_titles": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of titles that are filtered from incorrect values."
                        }
                    },
                    "required": ["valid_titles"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    titles = titles['title'].tolist()

    prompt = (
        f"Analyze the provided list or pandas dataframe of titles with the context of the website's theme: '{theme}'. "
        "Determine which titles are likely to be relevant to this theme and are proper article titles. "
        "Return a list of titles that are most likely to be relevant and valid for the specified theme, and filter out those that are not."
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a title analyzer specialized in matching titles to specific themes."},
            {"role": "user", "content": prompt + "\n\n" + "\n".join(titles)}
        ],
        tools=tools,
    )
    
    # print(completion.choices[0].finish_reason)
    
    if completion.choices[0].finish_reason != 'tool_calls':
        return titles
    try:
        res = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["valid_titles"]
        return res
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка при анализе результата: {e}")
        return False

      
      
