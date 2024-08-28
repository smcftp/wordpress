import asyncio
from typing import Tuple
import aiohttp
import pandas as pd
import numpy as np

from src.config import settings_pr

import random

from src.sentence_transformers import SentenceTransformer
from src.pinecone.grpc import PineconeGRPC as Pinecone
from src.pinecone import ServerlessSpec 

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")

# Преобразование данных в асинхронный контекст
async def process_semantic_similarity(df1: pd.DataFrame, df2: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    # Название индекса
    index_name = "semantic-comparison-index"
    
    # Инициализация Pinecone клиента
    pc = Pinecone(api_key=settings_pr.pinecone_api_key)

    # Спецификация для создания индекса
    spec = ServerlessSpec(
        cloud="aws",  # или другой облачный провайдер, если требуется
        region="us-east-1"  # или другой регион, если требуется
    )

    # Проверка существования индекса и его создание, если он не существует
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)

    index = pc.Index(index_name)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    list1 = df1['title'].tolist()
    list2 = df2['title'].tolist()

    # Преобразование текстов первого списка в векторы
    embeddings1 = model.encode(list1).tolist()

    # Вставка векторов первого списка в Pinecone
    for i, embed in enumerate(embeddings1):
        index.upsert(vectors=[(f"text1-{i}", embed)])

    # Преобразование текстов второго списка в векторы
    embeddings2 = model.encode(list2).tolist()

    to_remove_indices = set()

    # Поиск ближайших соседей для каждого вектора из второго списка
    for i, embed in enumerate(embeddings2):
        result = index.query(vector=embed, top_k=1, include_metadata=False)
        # print(f"Text from df1: '{list1[i]}'")
        for match in result['matches']:
            if match['score'] > similarity_threshold:
                list1_index = int(match['id'].split('-')[1])
                to_remove_indices.add(list1_index)
                # print(f"Most similar text from df1: '{list1[list1_index]}' with score {match['score']:.2f}")

    # Удаление элементов из df1, которые имеют семантические совпадения с df2
    filtered_df1 = df1.drop(to_remove_indices, errors='ignore').reset_index(drop=True)

    # Удаление индекса после завершения работы
    pc.delete_index(index_name)

    return filtered_df1
