import streamlit as st
from url_citation_number import *
from scrapearxiv import *
from semantic_api import *
from PDF_reader import *
from summarize import *
from save_pdf import *
import uuid
from pubmed import *
import concurrent.futures
import time
import numpy as np
from Rerank import *
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import add_script_run_ctx



start_time = time.time()
question = "What are the current limitations of long-read sequencing technologies in terms of accuracy and throughput?"
client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
)
keyword_prompt = f'Please extract 3 most relevant keywords based on the following question,' \
        f'You must respond with a list of strings in the following format: ["keyword 1", "keyword 2", "keyword 3"].' \
        f'Requirement: 1) The keywords should be used to search for relevant academic papers. ' \
        f'2) The keywords should summarize the main idea of the question. ' \
        f'3) The keywords can be supplements or extensions to the original question, not limited to the original question.' \
        f'Question below:{question}' 

keywords_content = get_keywords(keyword_prompt)

abstract_url = create_pubmed_url(keywords_content,type="abstract")
pubmed_url = create_pubmed_url(keywords_content,type="pubmed")



# 函数参数
start_page = 1
end_page = 5
ori_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/abstract_ori_result.csv"
pubmed_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/pubmed_ori_result.csv"
# 创建并行任务
with ThreadPoolExecutor(max_workers=2) as executor:
    future_abstract = executor.submit(scrape_format_abstract, start_page, end_page, abstract_url, ori_file_path)
    future_pubmed = executor.submit(scrape_format_pubmed_url, start_page, end_page, pubmed_url, pubmed_file_path)

    # 获取结果
    df1 = future_abstract.result()
    df2 = future_pubmed.result()

# 处理abstract数据集，去重并保存
df_unique = df1.drop_duplicates(subset='PMID', keep='first')
new_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/abstract_afterFusion_result.csv"
df_unique.to_csv(new_file_path, index=False)

# 合并两个DataFrame
merged_df = pd.merge(df_unique, df2, on='PMID', how='inner')
merged_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/merged_afterfusion.csv"
merged_df.to_csv(merged_file_path, index=False)

end_time = time.time()
print(end_time - start_time)



"""

import urllib.request 
import fitz 
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
import openai
from openai import OpenAI



def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings



def load_recommender(path, start_page=1):
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'



def generate_text(system_content, prompt):

    client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": "Here is some initial assistant message."},
            {"role": "user", "content": prompt}
        ],
        temperature=.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    if completion.choices:
        first_choice = completion.choices[0]
        message_content = first_choice.message.content
        return message_content
    else:
        print("No choices found.")

  
def generate_answer(prompt_type, question, model):
    topn_chunks = recommender(question)
    #print(topn_chunks)
    chunks = ""
    for c in topn_chunks:
        chunks += c + '\n\n'

    
    discussion_prompt = "Your task is to summarize the discussion section of an article provided. The discussion should be thorough, well-structured, informative, and provide in-depth analysis relevant to the current query. Use all the information provided to explore in depth The strengths and weaknesses of the approach to the topic are discussed with the aim of forming a discussion section in the report that focuses on a critical analysis of the results of the research topic while establishing a clear and valid personal perspective supported by the data." \
                    "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:"\
                    "-Start by synthesizing the information provided and pointing out interrelationships, trends, and potential contradictions between studies." \
                    "-Discuss unanswered questions, research gaps, and future research directions in the research field." \
                    "-Synthesize information from the literature review to propose a scientific approach to address the research question. Your approach should be clear, innovative, rigorous, valid and generalizable. This will be based on a deep understanding of the research question and its rationale, existing research and various entities."\
                    "You must develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                    "I am going to provide the related papers context, question as follows:" \
                    f"The research question is :{question}" \
                    f"The related search results is:{chunks}" \
                    "Then, following your review of the above content, Please generate a summary of the discussion part of this article in the format of" \
                    "Discussion:"

    introduction_prompt = "Your task is to summarize the introductory part of a paper provided. The summary should be comprehensive, well-structured, informative, and provide in-depth analysis relevant to the current query. Utilize all the information provided to summarize and expand, in-depth The importance and background significance of the topic are explored. The purpose is to provide an introduction to the report, highlighting the background significance and information background of the research topic."\
                "Preparation for the section of INTRODUCTION involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                "- First explain the background, importance, purpose and scope of the research topic based on the search results given." \
                "- Clearly define the scope of the research and elaborate on why the topic merits attention based on the search results given." \
                "- Develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                "I am going to provide the related papers context ,question as follows:" \
                f"The research question is :{question}" \
                f"The related search results is:{chunks}" \
                "Then, following your review of the above content, Please generate a summary of the introductory part of this article in the format of" \
                "Introduction:" 

    literature_prompt = "Your assignment is to summarize the literature review section of a paper provided. The summary should be thorough, well-structured, informative, and provide in-depth analysis relevant to the current query. Written using all the information provided A literature review not only answers the question but also delved into the literature on the topic. Its purpose is to summarize the literature review of the article, focusing on summarizing existing research related to the topic while establishing a clear and valid perspective supported by data. personal opinion." \
                    "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                    "- Begin by providing a detailed review and discussion of the literature mentioned in the above information, including experimental methods, experimental settings, experimental results, etc." \
                    "- Analyze the trends, successes, and shortcomings of existing research, which may include comparisons of theories, methodologies, or findings." \
                    "You must develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                    "I am going to provide the related papers context, question as follows:" \
                    f"The research question is: {question}" \
                    f"The related search results is:{chunks}" \
                    "Then, following your review of the above content, Please generate a summary of the literature review part of this article ,in the format of" \
                    "Literature Review:"


    introduction_system = "You are an AI assistant whose main goal is to summarize the research background and significance of the topic based on the given context, so as to help researchers write the Introduction section of the paper."
    literature_system = "You are an AI assistant whose main goal is to summarize the research status and progress of the topic based on the given context to help researchers write the Literature Review section of the paper."
    discussion_system = "You are an AI assistant whose main goal is to summarize the advantages and disadvantages of current research based on the given context to help researchers write the Literature Review section of their paper."

    answer_prompts = {
        "Introduction": introduction_prompt,
        "Literature": literature_prompt,
        "Discussion": discussion_prompt,
    }
    system_content = {
        "Introduction": introduction_system,
        "Literature": literature_system,
        "Discussion": discussion_system,
    }
    answer = generate_text(system_content[prompt_type],answer_prompts[prompt_type])
    return answer









import concurrent.futures

question = "Advancements and Challenges in Next-Generation Sequencing (NGS) Technologies for Genomic Analysis"
def process_document(url, prompts):
    # Assuming load_recommender and generate_answer require actual implementation
    load_recommender(url)  # Load the document into the recommender
    results = {}
    for section, prompt in prompts.items():
        # Assuming generate_answer is implemented to use models like GPT-3.5-turbo
        results[section] = generate_answer(section, prompt, model="gpt-3.5-turbo")
    return results

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
    future_to_url = {executor.submit(process_document, url, prompts): url for url in pdf_urls}
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























import requests

# 设置请求的 URL 和头部信息
url = 'https://langchain-3ff4ab2c9d.wolf.jina.ai/ask_url'
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

# 设置 POST 请求的数据体
data = {
    "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7784226/pdf/12248_2020_Article_532.pdf",
    "question": "Introduce the background information of the research topic of this article",
    "envs": {
        "OPENAI_API_KEY": "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"  # 替换为你的 OPENAI API 密钥
    }
}

# 发送 POST 请求
response = requests.post(url, json=data, headers=headers)
print(response)
# 输出响应内容
print(response.text)








sk-proj-MfwGsQ8GYms5LJoRn8MPT3BlbkFJarpxmEOKKDDEF03nyKBW













from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import os
import openai
import voyageai
from langchain_community.embeddings import JinaEmbeddings

from bs4 import BeautifulSoup
from langchain_voyageai import VoyageAIEmbeddings
os.environ["OPENAI_API_KEY"] = 'sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d'
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["VOYAGE_API_KEY"] = "pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE"
voyage_api_key = os.getenv('VOYAGE_API_KEY')

os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'


# 获取pdf文件内容
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text



# 拆分文本
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        # chunk_size=768,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return text_splitter.create_documents(chunks)


def get_contextual_retriever(file_path,prompt,embedding_model):

    documents = get_pdf_text(file_path)
    texts = get_text_chunks(documents)
    retriever = FAISS.from_documents(texts, embedding_model).as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(prompt)
    return docs



def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))



import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain.document_transformers import EmbeddingsRedundantFilter
#from summarize import *
# Function to process each URL and prompt
def process_pdf_url(url, prompts, embedding_model):
    context = {}
    for prompt_key, prompt_value in prompts.items():
        content = get_contextual_retriever(url, prompt_value, embedding_model)
        aggregated_text = " ".join(item.page_content for item in content)
        text = aggregated_text.replace("\n", " ")
        context[prompt_key] = text
    return context

# Measure the execution time
block5_start_time = time.time()

# Initialize embedding models and question
voyage_embeddings = VoyageAIEmbeddings(voyage_api_key="pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE", model="voyage-law-2")
jina_embeddings = JinaEmbeddings(jina_api_key="jina_e61db1cedc3c4b3fafe4e7632d8e00c5VW6PpZV4Y0wRoA4yKayeoW3JQcod", model_name="jina-embeddings-v2-base-en")
question = "How do epigenetic modifications influence the development and progression of neurodegenerative diseases like Alzheimer's?"

# List of PDF URLs
pdf_urls = ["/mnt/mydisk/wangyz/Research_agent/top_document/PMC6174532.pdf",
            "/mnt/mydisk/wangyz/Research_agent/top_document/PMC6380351.pdf"]

# Prompts dictionary
introduction = f"Question:{question} Based on the content of the above research questions, please introduce the background information and significance of the research on this topic, as well as the necessity and importance of this research topic, and the potential impact of this research topic."
literature = f"Question:{question} Based on the content of the above research questions, please introduce what experiments this article has set up around this topic, what the experimental process is, what the experimental results are, what methods are used to verify this topic, and why this method is proposed."
discussion = f"Question:{question} Based on the content of the above research questions, please explain the findings of the research topic, the significance of the results, and compare and contrast the research results with the results of other studies in related fields; Based on the findings and limitations of the current study, propose possible directions or suggestions for future research; Provide a summary of the entire study, emphasizing its contribution to the academic or practical field."
prompts = {
    "Introduction": introduction,
    "Literature": literature,
    "Discussion": discussion,
}



from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
context = defaultdict(dict)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=voyage_embeddings)
relevant_filter = EmbeddingsFilter(embeddings=voyage_embeddings, similarity_threshold=0.38)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)



# Use ThreadPoolExecutor to process PDFs concurrently
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_pdf, index, pdf_url): index for index, pdf_url in enumerate(pdf_urls)}

    for future in as_completed(futures):
        index, local_context = future.result()
        context[index] = local_context

 
# Print the extracted context for the first two documents
print("-------------------------------")
for i in range(2):  # Assuming at least two documents
    print(f"Document {i+1}:")
    print(f"Introduction: {context[i]['url']}\n")
    print(f"Introduction: {context[i]['Introduction']}\n")
    print(f"Literature: {context[i]['Literature']}\n")
    print(f"Discussion: {context[i]['Discussion']}\n")
    print("-------------------------------")  























# Initialize the paper and related paper details
paper = {"title": "aa", "abstract": "bb"}
relatedPaper = {"titles": "aa", "abstracts": "bb"}

# Use ThreadPoolExecutor to process the PDFs concurrently
context = {}
with ThreadPoolExecutor(max_workers=len(pdf_urls)) as executor:
    # Submit tasks to the executor
    futures = {executor.submit(process_pdf_url, url, prompts, voyage_embeddings, paper, relatedPaper): url for url in pdf_urls}
    
    # Collect the results as they complete
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        for key, value in result.items():
            if key not in context:
                context[key] = []
            context[key].append(value)
# Generate reports from the collected context
report_content = {}
for prompt_key, prompt_value in prompts.items():
    texts = context[prompt_key]
    combined_text = " ".join(texts)  # Combine all texts for the specific prompt
    report = generate_report(context=combined_text, question=question, prompt_type=prompt_key, paper=paper, relatedPaper=relatedPaper)
    report_content[prompt_key] = report

block5_end_time = time.time()
print(report_content)
print("内容读取完毕...")
print("这部分花费时间为：", block5_end_time - block5_start_time)




























block5_start_time = time.time()
print("按章节读取每一部分的内容...")
#读取每一部分内容
prompts_type = ["Introduction", "Literature", "Discussion"]
introduction = ""
literature = ""
discussion = ""


#ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=49aaa3d6f86cfdd65477f2d387677f37&Split=&Address=&isp=&poolnumber=0&qty=1"
proxies = []
for i in range(4):
    response = requests.get(ip_api_url)
    ip = response.text.strip()  # 去除末尾的多余字符
    proxies.append({"HTTP": f"HTTP://{ip}", "HTTPS": f"HTTP://{ip}"})
    print(f"添加代理: {ip}")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for i,pdf_url in enumerate(pdf_urls):
        chatpdf_proxy = proxies[i % len(proxies)]
        futures = [executor.submit(process_pdf, pdf_url, prompts_type, chatpdf_proxy)]

    for future in concurrent.futures.as_completed(futures):
        pdf_summaries = future.result()
        introduction += pdf_summaries['Introduction'] + "\n"
        literature += pdf_summaries['Literature'] + "\n"
        discussion += pdf_summaries['Discussion'] + "\n"

conclusion = introduction + literature + discussion

block5_end_time = time.time()
print("内容读取完毕...")
print("这部分花费时间为：",block5_end_time-block5_start_time)
"""