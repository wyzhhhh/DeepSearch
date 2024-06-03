import voyageai
import os
import time
import pandas as pd
import csv
import numpy as np
import requests
import time
from scipy.stats import rankdata
from impact_factor.core import Factor




#jina_api_key = "jina_a6f1d31fb5b34b3fa5ddfeff38105879lhkP5FSfnhZBJRNp0qFUwoS-XX5G"
#jina_api_key  = "jina_7a47404003bc4120b8f7afb3a3a689cc4JvVBsFnkcksJHa3LL7ICrrljMJG"

def Normalization(df,type):
    
    # 读取CSV文件
    #df = pd.read_csv(file_path)
    if type == "Cited Number":
        cited_numbers = df['Cited Number']
        ranked_citations = rankdata(cited_numbers, method='ordinal')  # 使用ordinal方法保证并列排名
        max_rank = ranked_citations.max()
        min_rank = ranked_citations.min()
        percentile_normalized = (ranked_citations - min_rank) / (max_rank - min_rank)

        df['cited_Normalized'] = percentile_normalized
    elif type == "impact_factor":
        cited_numbers = df['impact_factor']
        ranked_citations = rankdata(cited_numbers, method='ordinal')  # 使用ordinal方法保证并列排名
        max_rank = ranked_citations.max()
        min_rank = ranked_citations.min()
        percentile_normalized = (ranked_citations - min_rank) / (max_rank - min_rank)

        df['impact_Normalized'] = percentile_normalized

    return df


def impact_factor(journal_id):
    fa = Factor()
    response = fa.search(journal_id)
    if len(response) != 0:
        factor = response[0]['factor']
        print(f"impact factor for {journal_id}: {factor}")
        return factor
    else:
        print(f"impact factor for {journal_id}: {response}")
        return 0


def Jina_rerank(df,query,top_n,jina_api_key):

    #start_time = time.time()
    #csv_file_path = '/mnt/mydisk/wangyz/Research_agent/csv_download/merged_citations.csv'
    #df = pd.read_csv(file_path)
    if 'Abstract' in df.columns:
        documents = []
        for index, row in df.iterrows():
            title = row['Title'] if row['Title'] else 'No Title'
            abstract = row['Abstract'] if row['Abstract'] else 'No Abstract'
            documents.append(f"{title} {abstract}")
    else:
        print("The CSV file does not contain an 'abstract' column.")
   
    url = f"https://api.jina.ai/v1/rerank"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jina_api_key}"
    }

    data = {
        "model": "jina-reranker-v1-base-en",
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    response = requests.post(url, headers=headers, json=data)
    #print(response.json())
    response_text = response.json()
    results = response_text['results']
    new_data = []

    # 遍历results，获取对应的原始数据和相关性得分
    for result in results:
        index = result['index']
        relevance_score = result['relevance_score']
        # 根据index从原始df中找到对应的行
        original_data = df.iloc[index].to_dict()
        # 将相关性得分添加到原始数据中
        original_data['relevance_score'] = relevance_score
        new_data.append(original_data)

    new_df = pd.DataFrame(new_data)
    return new_df


def voyage_rerank(df,query,top_n,voyage_api_key):

    os.environ["VOYAGE_API_KEY"] = voyage_api_key
    voyage_api_key = os.getenv('VOYAGE_API_KEY')

    vo = voyageai.Client()
    documents = []
    if 'Abstract' in df.columns:
        for index, row in df.iterrows():
            title = row['Title'] if row['Title'] else 'No Title'
            abstract = row['Abstract'] if row['Abstract'] else 'No Abstract'
            documents.append(f"{title} {abstract}")
    else:
        print("The CSV file does not contain an 'abstract' column.")

    print(len(documents))
    if len(documents)>1000:

        middle_index = len(documents) // 2
        documents_part1 = documents[:middle_index]
        documents_part2 = documents[middle_index:]

        reranking_1 = vo.rerank(query, documents_part1, model="rerank-lite-1", top_k=top_n/2)
        reranking_2 = vo.rerank(query, documents_part2, model="rerank-lite-1", top_k=top_n/2)

        new_data = []
        # 处理第一部分的重新排名结果
        for r in reranking_1.results:
            index = r.index
            relevance_score = r.relevance_score
            original_data = df.iloc[index].to_dict()
            original_data['relevance_score'] = relevance_score
            new_data.append(original_data)
            print(f"Document: {r.document}")
            print(f"Relevance Score: {r.relevance_score}")

        # 处理第二部分的重新排名结果，注意索引需要加上第一部分的长度
        for r in reranking_2.results:
            index = r.index + middle_index  # 添加偏移量以匹配原始DataFrame中的索引
            relevance_score = r.relevance_score
            original_data = df.iloc[index].to_dict()
            original_data['relevance_score'] = relevance_score
            new_data.append(original_data)
            print(f"Document: {r.document}")
            print(f"Relevance Score: {r.relevance_score}")

        # 创建一个新的DataFrame，包含原始数据和相关性得分
        new_df = pd.DataFrame(new_data)
        return new_df
    
    else:
        new_data = []
        reranking = vo.rerank(query, documents, model="rerank-lite-1", top_k=top_n)
        for r in reranking.results:
            index = r.index
            relevance_score = r.relevance_score
            original_data = df.iloc[index].to_dict()
            original_data['relevance_score'] = relevance_score
            new_data.append(original_data)
            print(f"Document: {r.document}")
            print(f"Relevance Score: {r.relevance_score}")
        new_df = pd.DataFrame(new_data)
        return new_df




def local_rerank(df,query,top_n):


    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, Node, Document
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.service_context import ServiceContext
    import time
    import pandas as pd
    from langchain.embeddings import HuggingFaceBgeEmbeddings



    documents = []
    if 'Abstract' in df.columns:
        for index, row in df.iterrows():
            title = row['Title'] if row['Title'] else 'No Title'
            abstract = row['Abstract'] if row['Abstract'] else 'No Abstract'
            documents.append(f"{title} {abstract}")
    else:
        print("The CSV file does not contain an 'abstract' column.")


    embed_model = HuggingFaceBgeEmbeddings(model_name="/mnt/mydisk/wangyz/Reranker-evaluation/Model_path/bge-embedding-large/" , model_kwargs={'device': 'cuda'} , encode_kwargs={'normalize_embeddings': True})

    text_nodes = [TextNode(text=doc) for doc in documents]
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)

    # 使用 ServiceContext 创建 VectorStoreIndex
    vector_index = VectorStoreIndex(text_nodes,service_context=service_context)
    vector_retriever = VectorIndexRetriever(index=vector_index, service_context=service_context, similarity_top_k=top_n)
    retrieved_nodes = vector_retriever.retrieve(query)
    print(retrieved_nodes)
    retrieved_scores = {node.node.text: node.score for node in retrieved_nodes}
    new_data = []
    for index, row in df.iterrows():
        text = f"{row['Title']} {row['Abstract']}"
        if text in retrieved_scores:
            score = retrieved_scores[text]
            new_data.append(row.to_dict())
            new_data[-1]['relevance_score'] = score

    new_df = pd.DataFrame(new_data)
    return new_df






