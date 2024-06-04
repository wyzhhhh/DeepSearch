"""




from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
import pandas as pd
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
import numpy as np
import time


start_time = time.time()
reranker = FlagEmbeddingReranker(model="/mnt/mydisk/wangyz/Reranker-evaluation/Model_path/bge-reranker-large/", top_n=100)

# 指定CSV文件路径
csv_file_path = '/mnt/mydisk/wangyz/Research_agent/csv_download/merged_citations.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 确保'abstract'字段存在
if 'Abstract' in df.columns:
    # 将'abstract'字段的所有值存入一个列表中
    # 如果CSV文件中有多行数据，这将创建一个包含所有摘要的列表
    documents = df['Abstract'].tolist()

    
    # 打印结果
    #print(document)
else:
    print("The CSV file does not contain an 'abstract' column.")



nodes = [NodeWithScore(node=TextNode(text=doc)) for doc in documents]
query = "How do genetic variations in the APOE gene influence an individual's susceptibility to late-onset Alzheimer's disease, and what role do these variations play in the physiological mechanisms of the disease?"
query_bundle = QueryBundle(query_str=query)
ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
for i, node in enumerate(ranked_nodes):
    print(node.node.get_content(), "-> Score:", node.score)

end_time = time.time()

print("花时：",end_time-start_time)#125s











from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, Node, Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.service_context import ServiceContext
import time
import pandas as pd
from langchain.embeddings import HuggingFaceBgeEmbeddings

start_time = time.time()
csv_file_path = '/mnt/mydisk/wangyz/Research_agent/csv_download/merged_citations.csv'
df = pd.read_csv(csv_file_path)
if 'Abstract' in df.columns:
    documents = df['Abstract'].tolist()
else:
    print("The CSV file does not contain an 'abstract' column.")
query = "How do genetic variations in the APOE gene influence an individual's susceptibility to late-onset Alzheimer's disease, and what role do these variations play in the physiological mechanisms of the disease?"
embed_model = HuggingFaceBgeEmbeddings(model_name="/mnt/mydisk/wangyz/Reranker-evaluation/Model_path/bge-embedding-large/" , model_kwargs={'device': 'cuda'} , encode_kwargs={'normalize_embeddings': True})

text_nodes = [TextNode(text=doc) for doc in documents]
service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)

# 使用 ServiceContext 创建 VectorStoreIndex
vector_index = VectorStoreIndex(text_nodes,service_context=service_context)
vector_retriever = VectorIndexRetriever(index=vector_index, service_context=service_context, similarity_top_k=100)
retrieved_nodes = vector_retriever.retrieve(query)

# 打印出每个检索到的文档的文本和相似分数
print("Top 10 similar documents with their similarity scores:")
for node in retrieved_nodes:
    print(f"Document: {node.node.text}\nSimilarity Score: {node.score:.4f}")


end_time = time.time()
print("花时：",end_time-start_time)# 40s



vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

middle_index = len(documents) // 2
documents_part1 = documents[:middle_index]
documents_part2 = documents[middle_index:]

reranking_1 = vo.rerank(query, documents_part1, model="rerank-lite-1", top_k=50)
reranking_2 = vo.rerank(query, documents_part2, model="rerank-lite-1", top_k=50)
for r in reranking_1.results:
    print(f"Document: {r.document}")
    print(f"Relevance Score: {r.relevance_score}")

for r in reranking_2.results:
    print(f"Document: {r.document}")
    print(f"Relevance Score: {r.relevance_score}")



end_time = time.time()
print("花时：",end_time-start_time)# 5.36














import voyageai
import os
import time
import pandas as pd
import csv
import numpy as np
import requests
os.environ["VOYAGE_API_KEY"] = "pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE"
voyage_api_key = os.getenv('VOYAGE_API_KEY')


start_time = time.time()
csv_file_path = '/mnt/mydisk/wangyz/Research_agent/csv_download/merged_citations.csv'
df = pd.read_csv(csv_file_path)
if 'Abstract' in df.columns:
    documents = []
    for index, row in df.iterrows():
        title = row['Title'] if row['Title'] else 'No Title'
        abstract = row['Abstract'] if row['Abstract'] else 'No Abstract'
        documents.append(f"{title} {abstract}")
else:
    print("The CSV file does not contain an 'abstract' column.")
query = "How do genetic variations in the APOE gene influence an individual's susceptibility to late-onset Alzheimer's disease, and what role do these variations play in the physiological mechanisms of the disease?"




url = f"https://api.jina.ai/v1/rerank"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer jina_6c29330db7c243f489e11a2eec22846eNUAZ3USPzUnHIi5tuJkDApnffTIE"
}

data = {
    "model": "jina-reranker-v1-base-en",
    "query": query,
    "documents": documents,
    "top_n": 100
}

response = requests.post(url, headers=headers, json=data)
print(response.json())


end_time = time.time()
print("花时：",end_time-start_time)# 9.02



"""
from PDF_reader import *
import concurrent.futures

question = "Advancements and Challenges in Next-Generation Sequencing (NGS) Technologies for Genomic Analysis"


# PDF documents to process
pdf_urls = ["/mnt/mydisk/wangyz/Research_agent/top_document/PMC6174532.pdf", "/mnt/mydisk/wangyz/Research_agent/top_document/PMC6380351.pdf"]
introduction = f"Question:{question}" \
                "Based on the content of the above research questions,  please introduce the background information and significance of the research on this topic, as well as the necessity and importance of this research topic, and the potential impact of this research topic."

literature = f"Question:{question}" \
                "Based on the content of the above research questions,  please introduce what experiments this article has set up around this topic, what the experimental process is, what the experimental results are, what methods are used to verify this topic, and why this method is proposed."

discussion = f"Question:{question}" \
                "Based on the content of the above research questions,  Please explain the findings of the research topic, the significance of the results, and compare and contrast the research results with the results of other studies in related fields; Based on the findings and limitations of the current study, propose possible directions or suggestions for future research; Provide A summary of the entire study, emphasizing its contribution to the academic or practical field"


prompts = {
    "Introduction": introduction,
    "Literature": literature,
    "Discussion": discussion,
}
# Dictionary to store results
contents = []
recommender = SemanticSearch()

# Use ThreadPoolExecutor to handle tasks in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Map each URL to the process_document function
    future_to_url = {executor.submit(process_document, url, prompts,recommender): url for url in pdf_urls}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            contents.append(result)
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
for i in range(len(contents)):
    contents[i]['url'] = pdf_urls[i]



for i, ctx in enumerate(contents):
    print(f"The INTRODUCTION section of {ctx['url']} is: {ctx['Introduction']} ") 
    print(f"The DISCUSSION section of {ctx['url']} is: {ctx['Discussion']} ")
    print(f"The LITERATURE section of {ctx['url']} is: {ctx['Literature']} ")