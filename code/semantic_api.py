import requests
import time
import random
import csv
import pandas as pd
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import time
import random
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import contextlib
import io
import builtins
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pubmed import *
from PDF_reader import *





def get_abstract_from_ID(session, id, title, proxy=None):
    url = f"https://api.semanticscholar.org/v1/paper/{id}"
    headers = {'Connection': 'close'}
    
    try:
        response = session.get(url, headers=headers, proxies=proxy, timeout=10)
        while response.status_code == 403:
            time.sleep(500 + random.uniform(0, 500))  # 延迟等待，防止频繁请求
            response = session.get(url, headers=headers, proxies=proxy, timeout=10)  # 注意要再次传递代理

        if response.status_code == 200:
            data = response.json()
            abstract = data.get("abstract", "No abstract available")
            print(id)
            print(title)
            print(abstract)
            print("--")
        else:
            abstract = "No abstract available due to error: " + str(response.status_code)
    except Exception as e:
        abstract = "No abstract available due to exception: " + str(e)

    return id, title, abstract




def get_references_citations(session,citation_path,references_path,proxy=None,arxiv_id=None,doi=None):

    if arxiv_id:
        url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    if doi:
        url = f"https://api.semanticscholar.org/v1/paper/DOI:{doi}"

        """
        pmcid = get_id_from_doi(doi, output_type="pmcid")
        if pmcid == None:
            pmid = get_id_from_doi(doi, output_type="pmid")
            if pmid:
                id = pmid
        else:
            id = pmcid

        """
    headers = {
        'Connection': 'close'  # 设置为关闭长连接
    }

    #response = requests.get(url, headers=headers)
    response = session.get(url, headers=headers, proxies=proxy)
    while response.status_code == 403:
        time.sleep(500 + random.uniform(0, 500))
        response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        data = response.json()
        # 检查是否有引用文献
        if "references" in data and len(data["references"]) > 0:
            references_list=[]
            with open(references_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # 写入标题行
                writer.writerow(["arxivId", "Title", "Year", "Semantic Scholar ID", "URL"])
                # 写入引用文献的信息
                for reference in data["references"]:
                    """
                    with contextlib.redirect_stdout(io.StringIO()) as f:
                        print(f"arxivId: {reference.get('arxivId', 'No title available')}")
                        print(f"Title: {reference.get('title', 'No title available')}")
                        print(f"Year: {reference.get('year', 'No year available')}")
                        print(f"Semantic Scholar ID: {reference.get('paperId', 'No ID available')}")
                        print(f"URL: {reference.get('url', 'No year available')}")
                        # 每次打印后立即发送输出内容到前端
                        output = f.getvalue()
                        emit('update_output', {'data': output})
                        f.truncate(0)
                        f.seek(0)

                    """
                    
                    print(f"arxivId: {reference.get('arxivId', 'No title available')}")
                    print(f"Title: {reference.get('title', 'No title available')}")
                    print(f"Year: {reference.get('year', 'No year available')}")
                    print(f"Semantic Scholar ID: {reference.get('paperId', 'No ID available')}")
                    references_list.append(reference.get('paperId', 'No ID available'))
                    print(f"URL: {reference.get('url', 'No year available')}")
                    print("-----")

                    #references_list.append(reference.get('paperId', 'No ID available'))
                    writer.writerow([
                        reference.get('arxivId', 'No title available'),
                        reference.get('title', 'No title available'),
                        reference.get('year', 'No year available'),
                        reference.get('paperId', 'No ID available'),
                        reference.get('url', 'No URL available')
                    ])

        if "citations" in data and len(data["citations"]) > 0:
            citation_list=[]
            with open(citation_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # 写入标题行
                writer.writerow(["DOI", "Title", "Year","Semantic Scholar ID", "URL","Intent","isInfluential"])
                # 写入引用文献的信息
                for citations in data["citations"]:
                    """
                    with contextlib.redirect_stdout(io.StringIO()) as f:
                        print(f"arxivId: {citations.get('arxivId', 'No title available')}")
                        print(f"Title: {citations.get('title', 'No title available')}")
                        print(f"Year: {citations.get('year', 'No year available')}")
                        print(f"Semantic Scholar ID: {citations.get('paperId', 'No ID available')}")
                        print(f"URL: {citations.get('url', 'No year available')}")
                        print("-----")
                        # 每次打印后立即发送输出内容到前端
                        output = f.getvalue()
                        emit('update_output', {'data': output})
                        f.truncate(0)
                        f.seek(0)


                    """
                    
                    
                    print(f"Title: {citations.get('title', 'No title available')}")
                    print(f"Year: {citations.get('year', 'No year available')}")
                    print(f"Semantic Scholar ID: {citations.get('paperId', 'No ID available')}")
                    print(f"URL: {citations.get('url', 'No year available')}")
                    print(f"doi: {citations.get('doi', 'No doi available')}")
                    print("-----")


                    intent_list = citations.get('intent', [])
                    intent_value = ','.join(intent_list) if intent_list else ''

                    citation_list.append(citations.get('paperId', 'No ID available'))
                    writer.writerow([
                        citations.get('doi', 'No arxivId available'),
                        citations.get('title', 'No title available'),
                        citations.get('year', 'No year available'),
                        citations.get('paperId', 'No paperID available'),
                        citations.get('url', 'No URL available'),
                        intent_value,
                        citations.get('isInfluential','No isInfluential available')

                    ])
    response.close()




def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    length1=len(set1)
    length2=len(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0  # 避免除以零的错误

    return intersection/length2

def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session



def get_similarity(session, paper_id, origi_references_list, proxy):
    
    headers = {
        'Connection': 'close'
    }

    
    url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
    print(url)
    try:
        response = session.get(url, headers=headers, proxies=proxy,)
        response.raise_for_status()
        data = response.json()
        references_list = [ref.get('paperId', 'No ID available') for ref in data.get("references", [])]
        similarity = jaccard_similarity(origi_references_list, references_list)
        cited_number = len(data["citations"])
        # 将相似度分数添加到DataFrame中

        print(f"Similarity for {paper_id}: {similarity}")
        print(f"cited number for {paper_id}: {cited_number}")
        response.close()
        return paper_id,similarity,cited_number
    except requests.exceptions.RequestException as e:

        print(f"Error with paper ID {paper_id}: {e}")
        return paper_id, None, None  # 如果出错，返回None


        """
        
        with contextlib.redirect_stdout(io.StringIO()) as f:
            print(f"Similarity for {paperid}: {similarity}")
            # 每次打印后立即发送输出内容到前端
            output = f.getvalue()
            emit('update_output', {'data': output})
            f.truncate(0)
            f.seek(0)
        
        """


    






"""
def get_similarity(paperid,origi_references_list,filepath):

    url = f"https://api.semanticscholar.org/v1/paper/{paperid}"
    headers = {
        'Connection': 'close'  # 设置为关闭长连接
    }
    response = requests.get(url, headers=headers)

    while response.status_code == 403:
        time.sleep(500 + random.uniform(0, 500))
        response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        data = response.json()
        with open('data_similarity.json', 'w') as json_file:
            json.dump(data, json_file)

    response.close()


    with open('data_similarity.json', 'r') as json_file:
        data = json.load(json_file)

    if "references" in data and len(data["references"]) > 0:
        references_list=[]
        for reference in data["references"]:
            #print(f"arxivId: {reference.get('arxivId', 'No title available')}")
            #print(f"Title: {reference.get('title', 'No title available')}")
            #print(f"Year: {reference.get('year', 'No year available')}")
            #print(f"Semantic Scholar ID: {reference.get('paperId', 'No ID available')}")
            references_list.append(reference.get('paperId', 'No ID available'))
            #print(f"文章 {reference.get('title', 'No title available')} 已添加进references_list")
            #print(f"URL: {reference.get('url', 'No year available')}")
            #print("-----")

    similarity = jaccard_similarity(origi_references_list, references_list)
    with open(filepath, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            data["title"],
            data["abstract"],
            data["arxivId"],
            data["url"],
            data["doi"],
            similarity
        ])
"""







"""
df = pd.read_csv('../BM25/citation_arxiv.csv')
# 确保'Citation Number'列的数据类型为整数
df['Citation Number'] = pd.to_numeric(df['Citation Number'], errors='coerce')
filtered_df = df[df['Citation Number'] > 100]
for index, row in filtered_df.iterrows():
    arxiv_number = row['ArXiv Number']
    # 创建一个基于ArXiv Number的文件名
    references_filename = f"{arxiv_number}_references.csv"
    citations_filename = f"{arxiv_number}_citations.csv"
    print("arxiv number:",arxiv_number)
    get_references_citations(arxiv_number,citations_filename,references_filename)
    
    
    
    
  
arxiv_number=2312.10997
references_filename = "2312.10997_references.csv"
citations_filename = "2312.10997_citations.csv"
get_references_citations(arxiv_number,citations_filename,references_filename)    
    



df_origi = pd.read_csv('2312.10997_references.csv')
origi_references_list = df_origi['Semantic Scholar ID'].tolist()

df_citations = pd.read_csv('2312.10997_citations.csv')
filepath='output_similarity.csv'

with open(filepath, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(
    ["Title", "Abstract", "ArXiv ID", "URL", "交集/原始", "交集/本文", "交集/并集"])

for paperid in df_citations['Semantic Scholar ID']:
    get_similarity(paperid, origi_references_list, filepath)



import requests
import json

# 设置 API 的基础 URL
base_url = "https://api.semanticscholar.org/recommendations/v1"

# 指定要请求的论文推荐的 API 路径和参数
paper_id = "075f320d8e82673b51204a768d831a17f9999c02"
path = f"/papers/forpaper/{paper_id}"
params = {
    "limit": 10,  # 请求返回的推荐论文数量
    "fields": "title,authors,year"  # 请求返回的字段
}

# 发起 GET 请求
response = requests.get(f"{base_url}{path}", params=params)

# 检查请求是否成功
if response.status_code == 200:
    # 解析响应内容
    recommendations = response.json()
    print(json.dumps(recommendations, indent=2))
else:
    print(f"Error: {response.status_code}")

"""
