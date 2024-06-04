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
from summarize import *
from bs4 import BeautifulSoup
from langchain_voyageai import VoyageAIEmbeddings
os.environ["OPENAI_API_KEY"] = 'sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d'
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["VOYAGE_API_KEY"] = "pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE"
voyage_api_key = os.getenv('VOYAGE_API_KEY')

os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'


from pubmed import *
x_api_key='sec_wIIMRcjND18eu3DKWjYg3nngYaz9hta3'


import urllib.request 
import fitz 
import re
import numpy as np
import tensorflow_hub as hub
import gradio as gr
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI


client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
)

def download_pdf(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 确保网址正确响应
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("PDF文件已成功下载到:", save_path)
    except requests.RequestException as e:
        print("下载过程中出现问题:", e)


def get_summarize(url, prompt_type, proxy,filepath=None):

    prompts = {
        "Introduction": "Please tell us about the background of this article, as well as the history of its development, which was used to write the Introduction section of the paper",
        "Literature": "Please tell us about the purpose, setup, process, and results of this article",
        "Discussion": "Please explain the advantages and disadvantages of this article and propose ideas to solve these disadvantages.",
    }
    prompt = prompts[prompt_type]

    if filepath:
        files = [('file', ('file', open(filepath, 'rb'), 'application/octet-stream'))]
        headers = {'x-api-key': x_api_key, 'Connection': 'close'}
        response = requests.post('https://api.chatpdf.com/v1/sources/add-file', proxies=proxy,headers=headers, files=files)
    else:
        headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
        data = {'url': url}
        response = requests.post('https://api.chatpdf.com/v1/sources/add-url', proxies=proxy,headers=headers, json=data)

    if response.status_code == 200:
        source_id = response.json()['sourceId']
        print("source_id:",source_id)
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)
        return
    response.close()

    data = {
        'sourceId': source_id,
        'messages': [{'role': "user", 'content': prompt}]
    }
    response = requests.post('https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

    if response.status_code == 200:
        summary = response.json()['content'] + "\nThe source URL of above content is:" + url
        print(summary)
        return summary
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)

    response.close()


def extract_more(csv_path, top_n):
    data = pd.read_csv(csv_path, encoding='ISO-8859-1')
    top_data = data.sort_values(by='final_score', ascending=False)

    DOIs = top_data['DOI'].head(top_n).tolist()
    abstracts = top_data['Abstract'].head(top_n).tolist()
    titles = top_data['Title'].head(top_n).tolist()
    impact_factors = top_data['impact_factor'].head(top_n).tolist()
    
    DOIs = [doi.replace("\n", " ") for doi in DOIs]
    abstracts = [abstract.replace("\n", " ") for abstract in abstracts]
    titles = [title.replace("\n", " ") for title in titles]
    impact_factors = [str(impact_factor).replace("\n", " ") for impact_factor in impact_factors]
   

    return DOIs, abstracts, titles, impact_factors


def extract_pdf_similarity(csv_path, type,top_n):
    data = pd.read_csv(csv_path, encoding='ISO-8859-1')
    top_data = data.sort_values(by='final_score', ascending=False)
    pdf_urls = []
    abstracts = []
    titles = []
    relatedpapers = {
        'titles': [],
        'abstracts': []
    }

    if type == "arxiv":
        for _, row in top_data.iterrows():
            if len(pdf_urls) >= top_n:
                break
            pdf_url = f"https://arxiv.org/pdf/{row['arxivId']}.pdf"
            pdf_urls.append(pdf_url)
            abstracts.append(row['Abstract'])

    elif type == "pubmed":
        for _, row in top_data.iterrows():
            if len(pdf_urls) >= top_n:
                break



            pmcid = get_id_from_doi(row['DOI'],output_type="pmcid")
            if pmcid:
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                pdf_urls.append(pdf_url)
                abstracts.append(row['Abstract'])

                relatedpapers['titles'].append(row['Title'])
                relatedpapers['abstracts'].append(row['Abstract'])



            else:
                #print(f"Can't find PMCID for DOI: {row['DOI']}")
                pmid = get_id_from_doi(row['DOI'], output_type="pmid")
                if pmid:
                    # 通过爬取网页的内容来得到PMCID
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    response = requests.get(url)
                    # 检查请求是否成功
                    if response.status_code == 200:
                        # 解析页面内容
                        soup = BeautifulSoup(response.content, 'html.parser')
                        pmc_id_element = soup.find('a', {'href': lambda x: x and 'pmc/articles/pmc' in x})
                        # 提取PMC ID
                        if pmc_id_element:
                            pmcid = pmc_id_element.text.strip()
                            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                            pdf_urls.append(pdf_url)
                            abstracts.append(row['Abstract'])
                            titles.append(row['Title'])
                            relatedpapers['titles'].append(row['Title'])
                            relatedpapers['abstracts'].append(row['Abstract'])

                        else:
                            print("PMC ID not found.")
                    else:
                        print(f"Failed to retrieve the page. Status code: {response.status_code}")





        return pdf_urls, abstracts, titles, relatedpapers


def extract_pdf_url(pmid):

    title = ""
    abstract = ""
    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功

    # 解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    #提取title
    title_meta = soup.find('meta', {'name': 'citation_title'})
    if title_meta:
        title=title_meta.get('content')
    else:
        print("Title content not found.")


    # 定位abstract内容
    abstract_div = soup.find('div', {'class': 'abstract-content selected', 'id': 'eng-abstract'})
    if abstract_div:
        # 提取abstract中的每一段内容
        abstract_paragraphs = abstract_div.find_all('p')
        abstract_content = []
        for para in abstract_paragraphs:
            sub_title = para.find('strong', {'class': 'sub-title'})
            if sub_title:
                # 合并sub-title和其余内容
                combined_text = f"{sub_title.text.strip()}: {para.get_text().replace(sub_title.text, '').strip()}"
                abstract_content.append(combined_text)
            else:
                abstract_content.append(para.get_text().strip())

        abstract = " ".join(abstract_content)
    else:
        print("Abstract content not found.")

    return title,abstract


def process_pdf(pdf_url, prompts_type, proxy):
    parts = pdf_url.split('/')
    pmc_id = parts[5]
    pdf_save_path = f"/home/user/Deepsearch/top_papers/{pmc_id}.pdf"
    download_pdf(pdf_url, pdf_save_path)  # 下载pdf文件

    pdf_summaries = {'Introduction': '', 'Literature': '', 'Discussion': ''}

    for prompt_type in prompts_type:
        start = pdf_url.rfind('/') + 1
        end = pdf_url.rfind('.pdf')
        arxiv_number = pdf_url[start:end]
        text = get_summarize(url=pdf_url, prompt_type=prompt_type, proxy=proxy, filepath=pdf_save_path)
        pdf_summaries[prompt_type] = text

    return pdf_summaries


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


def process_pdf_url(index, pdf_url,voyage_embeddings,pipeline_compressor,prompts):

    match = re.search(r'/([^/]+)\.pdf$', pdf_url)
    local_context = {'url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/{match.group(1)}/pdf/"}
    try:
        # Extract text from the PDF
        documents = get_pdf_text(pdf_url)
        texts = get_text_chunks(documents)

        # Set up the FAISS retriever with embedded texts
        retriever = FAISS.from_documents(texts, voyage_embeddings).as_retriever(search_kwargs={"k": 3})

        # Set up contextual compression with a base compressor and the FAISS retriever
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

        # Retrieve relevant documents for each prompt and store them in the context dictionary
        for prompt_key, prompt_value in prompts.items():
            relevant_docs = contextual_retriever.get_relevant_documents(prompt_value)
            combined_content = " ".join(doc.page_content for doc in relevant_docs)
            local_context[prompt_key] = combined_content

    except Exception as e:
        print(f"Error processing PDF at index {index} with URL {pdf_url}: {str(e)}")

    return index, local_context








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



def load_recommender(path, recommender, start_page=1):

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
        temperature=.7,
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

  
def generate_answer(recommender,prompt_type, question, model):
    
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



def process_document(url, prompts,recommender):
    
    load_recommender(url,recommender)  
    results = {}
    for section, prompt in prompts.items():
        results[section] = generate_answer(recommender,section, prompt, model="gpt-3.5-turbo")
    return results

"""
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
recommender = SemanticSearch()
prompts = {
    "Introduction": "a",
    "Literature": "b",
    "Discussion": "c",
}

combined_text = []
recommender = SemanticSearch()
pdf_urls = ['https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8114529/pdf/', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8675181/pdf/', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8762559/pdf/', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8748638/pdf/', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10256258/pdf/']
pdfs = []
for i,pdf_url in enumerate(pdf_urls):
    parts = pdf_url.split('/')
    pmc_id = parts[5]
    pdf_save_path = f"/mnt/mydisk/wangyz/Research_agent/top_document/{pmc_id}.pdf"
    pdfs.append(pdf_save_path)

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {executor.submit(process_document, url, prompts,recommender): url for url in pdfs}
    total_pdfs = len(future_to_url)
    completed = 0
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            combined_text.append(result)
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
        completed += 1



for i in range(len(combined_text)):
    combined_text[i]['url'] = pdf_urls[i]



for i, ctx in enumerate(combined_text):
    print(f"The INTRODUCTION section of {ctx['url']} is below:\n {ctx['Introduction']} ")
    print(f"The DISCUSSION section of {ctx['url']} is below:\n {ctx['Discussion']} ") 
    print(f"The LITERATURE section of {ctx['url']} is below:\n {ctx['Literature']} ") 
"""