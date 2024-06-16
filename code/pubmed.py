from openai import OpenAI
import openai
from scrapearxiv import scrape_arxiv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import xml.etree.ElementTree as ET
import time
import random
client = OpenAI(
    #base_url="https://kapkey.chatgptapi.org.cn/v1",
    #api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
    
    base_url = "https://api.xiaoai.plus/v1",
    api_key = "sk-hz4C02ZEZUjbkk0aE92028468246454793Bc6649F0Bb1b9e"

    #base_url = "https://api.closeai-proxy.xyz/v1",
    #api_key = "sk-v1Y3L4qrAPFGKMIAJ4wZ5H8eJxAuH97GYnA4iFm0pwqlKaJx"

)

def create_pubmed_nested_query(keywords):
    if not keywords:
        return ""
    elif len(keywords) == 1:
        return f"%28{keywords[0]}%29"
    else:
        mid = len(keywords) // 2
        left_part = create_pubmed_nested_query(keywords[:mid])
        right_part = create_pubmed_nested_query(keywords[mid:])
        return f"%28{left_part}+OR+{right_part}%29"

def create_pubmed_url(keywords,type):
    # 生成嵌套的查询表达式
    nested_query = create_pubmed_nested_query(keywords)
    # 返回完整的PubMed查询URL
    if type == "abstract":
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={nested_query}&filter=simsearch1.fha&filter=years.2015-2025&format=abstract&size=200"
    elif type == "pubmed":
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={nested_query}&filter=years.2015-2025&format=pubmed&size=200"
    else:
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={nested_query}&filter=years.2015-2025&size=200"



def get_pubmed_citation_Number(doi,proxy):


    url = f"https://api.semanticscholar.org/v1/paper/DOI:{doi}"
    headers = {
        'Connection': 'close'  # 设置为关闭长连接
    }
    response = requests.get(url,proxies=proxy,headers=headers)
    if response.status_code == 404:
        print(f"请求失败，状态码：{response.status_code}")
        return 0

    while response.status_code == 403:
        time.sleep(500 + random.uniform(0, 500))
        response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "citations" in data and len(data["citations"]) > 0:
            print("the total citations number is :", len(data["citations"]))
            return len(data["citations"])
        else:
            print("the total citations number is :", len(data["citations"]))
            return 0

    print(f"citations number请求失败，状态码：{response.status_code}")
    response.close()
    return 0

def parse_xml_for_id(xml_data,output_type):
    root = ET.fromstring(xml_data)
    record = root.find('.//record')
    if record is not None and 'pmcid' in record.attrib and output_type == "pmcid":
        pmcid=record.attrib['pmcid']
        return pmcid
    if record is not None and 'pmid' in record.attrib and output_type == "pmid":
        pmid=record.attrib['pmid']
        return pmid
    if record is not None and 'pmid' in record.attrib and output_type == "doi":
        doi=record.attrib['doi']
        return doi
    else:
        return

def get_id_from_doi(doi,output_type):
    # 构建查询URL
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        'ids': doi,
        'format': 'xml'  # 指定返回格式为XML
    }

    # 发送GET请求
    response = requests.get(url, params=params)

    # 检查响应状态码
    if response.status_code == 200:
        if response.text:
            if output_type == "pmcid":
                pmcid = parse_xml_for_id(response.text,output_type)
                print(f"Successfully retrieved PMCID {pmcid} from ", doi)
                return pmcid
            if output_type == "pmid":
                pmid = parse_xml_for_id(response.text,output_type)
                print(f"Successfully retrieved PMID {pmid} from ", doi)
                return pmid
        #print("No data to parse.")
        return None
    else:
        #print(f"Failed to retrieve data: Status code {response.status_code}")
        return None

def get_pmcid_from_pmid(pmid,type=None):
    # 构建查询URL
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        'ids': pmid,
        'format': 'xml'  # 指定返回格式为XML
    }

    # 发送GET请求
    response = requests.get(url, params=params)
    
    # 检查响应状态码
    if response.status_code == 200:
        if response.text:
            print(response.text)
            pmcid = parse_xml_for_id(response.text,output_type="pmcid")
            doi = parse_xml_for_id(response.text,output_type="doi")
            if type == "doi":
                return doi
            return pmcid
        else:
            #print("No data to parse.")
            return None
    else:
        #print(f"Failed to retrieve data: Status code {response.status_code}")
        return None


