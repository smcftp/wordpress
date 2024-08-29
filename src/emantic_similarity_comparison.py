import asyncio
from typing import Tuple
import aiohttp
import pandas as pd
import numpy as np

from src.config import settings_pr

import random

from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec 

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")

# Преобразование данных в асинхронный контекст
async def process_semantic_similarity(df1: pd.DataFrame, df2: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    try:
        # Название индекса
        index_name = "semantic-comparison-index"
        
        # Инициализация Pinecone клиента
        try:
            pc = Pinecone(api_key=settings_pr.pinecone_api_key)
        except Exception as e:
            print(f"Ошибка инициализации Pinecone клиента: {e}")
            return pd.DataFrame()

        # Спецификация для создания индекса
        spec = ServerlessSpec(
            cloud="aws",  # или другой облачный провайдер, если требуется
            region="us-east-1"  # или другой регион, если требуется
        )

        # Проверка существования индекса и его создание, если он не существует
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)
        except Exception as e:
            print(f"Ошибка при создании или проверке индекса Pinecone: {e}")
            return pd.DataFrame()

        try:
            index = pc.Index(index_name)
        except Exception as e:
            print(f"Ошибка при создании объекта индекса Pinecone: {e}")
            return pd.DataFrame()

        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Ошибка при загрузке модели SentenceTransformer: {e}")
            return pd.DataFrame()

        list1 = df1['title'].tolist()
        list2 = df2['title'].tolist()

        # Преобразование текстов первого списка в векторы
        try:
            embeddings1 = model.encode(list1).tolist()
        except Exception as e:
            print(f"Ошибка при преобразовании текстов df1 в векторы: {e}")
            return pd.DataFrame()

        # Вставка векторов первого списка в Pinecone
        try:
            for i, embed in enumerate(embeddings1):
                index.upsert(vectors=[(f"text1-{i}", embed)])
        except Exception as e:
            print(f"Ошибка при вставке векторов в Pinecone: {e}")
            return pd.DataFrame()

        # Преобразование текстов второго списка в векторы
        try:
            embeddings2 = model.encode(list2).tolist()
        except Exception as e:
            print(f"Ошибка при преобразовании текстов df2 в векторы: {e}")
            return pd.DataFrame()

        to_remove_indices = set()

        # Поиск ближайших соседей для каждого вектора из второго списка
        try:
            for i, embed in enumerate(embeddings2):
                result = index.query(vector=embed, top_k=1, include_metadata=False)
                for match in result['matches']:
                    if match['score'] > similarity_threshold:
                        list1_index = int(match['id'].split('-')[1])
                        to_remove_indices.add(list1_index)
        except Exception as e:
            print(f"Ошибка при поиске ближайших соседей в Pinecone: {e}")
            return pd.DataFrame()

        # Удаление элементов из df1, которые имеют семантические совпадения с df2
        try:
            filtered_df1 = df1.drop(to_remove_indices, errors='ignore').reset_index(drop=True)
        except Exception as e:
            print(f"Ошибка при удалении совпадающих элементов из df1: {e}")
            return pd.DataFrame()

        # Удаление индекса после завершения работы
        try:
            pc.delete_index(index_name)
        except Exception as e:
            print(f"Ошибка при удалении индекса Pinecone: {e}")

        return filtered_df1

    except Exception as e:
        print(f"Неизвестная ошибка в функции process_semantic_similarity: {e}")
        return pd.DataFrame()

