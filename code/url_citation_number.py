from openai import OpenAI
import openai
#from scrapearxiv import scrape_arxiv
import pandas as pd
import requests
import time
import random


client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
)
def get_subqueris(prompt):

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
    )

    if completion.choices:
        # 提取第一个选项
        first_choice = completion.choices[0]
        # 提取该选项的消息内容
        message_content = first_choice.message.content
        print(message_content)
        # 打印内容
        sub_queries = []

        # 使用逗号和句号分割问题
        questions = message_content.split(',')

        # 将分割后的问题加入 sub_queries 列表
        for question in questions:
            question = question.strip().strip('["').strip('"]')
            if question:
                sub_queries.append(question)

        return sub_queries
    else:
        print("No choices found.")

def get_keywords(prompt):

    system_content = "You are an AI assistant whose main goal is to summarize three keywords based on a given question that will be used to help researchers search for relevant literature."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": prompt}],
    )

    if completion.choices:
        # 提取第一个选项
        first_choice = completion.choices[0]
        # 提取该选项的消息内容
        message_content = first_choice.message.content
        # 打印内容
        key_words = []

        # 使用逗号和句号分割问题
        keywords = message_content.split(',')
        for keyword in keywords:
            keyword = keyword.strip().strip('["').strip('"]')
            if keyword:
                key_words.append(keyword)

        return key_words
    else:
        print("No choices found.")

def get_url(key_words):
    base_url = 'https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&'
    search_terms = ''
    for i, keyword in enumerate(key_words):
        # 添加逻辑运算符（AND/OR）和字段
        if i >= 0:
            search_terms += f'&terms-{i}-operator=OR'
            search_terms += f'&terms-{i}-term={"+".join(keyword.split())}'
            search_terms += f'&terms-{i}-field=all'

    search_terms += f'&classification-computer_science=y&classification-economics=y&classification-eess=y&classification-mathematics=y&classification-physics=y&classification-physics_archives=all&classification-q_biology=y&classification-q_finance=y&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=200&order='

    # 将计算出的搜索条件添加到基础 URL 中
    full_url = base_url + search_terms
    print(full_url)
    return full_url

def get_semanticID(arxiv_id):
    url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    # 发起请求
    response = requests.get(url)

    while response.status_code == 403:
        print(f"Semantic Scholar ID请求失败，状态码：{response.status_code}")
        time.sleep(500 + random.uniform(0, 500))
        response = requests.get(url)
    if response.status_code == 404:
        print(f"Semantic Scholar ID请求失败，状态码：{response.status_code}")
        return 0
    if response.status_code == 200:
        data = response.json()
        semantic_scholar_id = data.get("paperId")
        print(f"Semantic Scholar ID: {semantic_scholar_id}")
        return semantic_scholar_id


"""

def get_arxiv_citation_Number(arxiv_id):


    url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    headers = {
        'Connection': 'close'  # 设置为关闭长连接
    }
    response = requests.get(url,headers=headers)
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

"""




def get_pubmed_citation_Number(doi):
    url = f"https://api.semanticscholar.org/v1/paper/DOI:{doi}"
    headers = {
        'Connection': 'close'  # 设置为关闭长连接
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        print(f"请求失败，状态码：{response.status_code}")
        return 0

    retries = 0
    while response.status_code == 403 or response.status_code == 429:
        retries += 1
        if retries > 5:
            print(f"请求失败，重试次数过多，状态码：{response.status_code}")
            return 0
        sleep_time = min(2 ** retries + random.uniform(0, 1), 60)  # 指数退避，最多延迟60秒
        print(f"状态码：{response.status_code}，等待{sleep_time}秒后重试...")
        time.sleep(sleep_time)
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        time.sleep(random.uniform(0, 15))
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


def new_get_url(key_words):
    base_url = 'https://arxiv.org/search/advanced?advanced=&'
    search_terms = ''
    for i, keyword in enumerate(key_words):
        formatted_keyword = keyword.replace(' ', '+').replace('(', '%28').replace(')', '%29')
        print("--------------------------------------")
        print(formatted_keyword)
        if i == 0:
            operator = 'AND'
        else:
            operator = 'OR'
        search_terms += f'terms-{i}-operator={operator}&'
        search_terms += f'terms-{i}-term={formatted_keyword}&'
        search_terms += f'terms-{i}-field=all&'

    search_terms += f'classification-computer_science=y&classification-economics=y&classification-eess=y&classification-mathematics=y&classification-physics=y&classification-physics_archives=all&classification-q_biology=y&classification-q_finance=y&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=200&order='

    # Combine the base URL and search parameters
    full_url = base_url + search_terms
    print(full_url)
    return full_url

def extract_content(text,keyword):
    text = text.replace('\n', ' ')
    lower_text = text.lower()

    problem_start = lower_text.find(keyword) + len(keyword)
    rationale_start = lower_text.find("rationale:") + len("rationale:")

    problem_end = lower_text.find("rationale:")
    rationale_end = len(lower_text)

    problem = text[problem_start:problem_end].strip()
    rationale = text[rationale_start:rationale_end].strip()

    return problem, rationale



# Example usage

"""

#question=input("please input your question:")
question="Please give specific academic research reports on the intelligent answer systems in various vertical fields that Retrieval-Augment-Generate(RAG) combined with LLM and knowledge graph construction."

subquery_prompt = f'Based on the following question, propose three additional questions with the following requirements.' \
                  f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3"].' \
                  f'Requirement:1)The new questions should be used for relevant academic research searches. ' \
                  f'2)The new questions should supplement and expand on the technical content or approaches mentioned in the original question.' \
                  f'3)The new questions should be as detailed as possible in their description.' \
                  f'Question below: {question}' \



sub_queries=get_subqueris(subquery_prompt)
sub_queries.append(question)
print(sub_queries)

key_words=[]
for item in sub_queries:

    keyword_prompt = f'Please extract 3 most relevant keywords based on the following question,' \
                     f'You must respond with a list of strings in the following format: ["keyword 1", "keyword 2", "keyword 3"].' \
                     f'Requirement: 1) The keywords should be used to search for relevant academic papers. ' \
                     f'2) The keywords should summarize the main idea of the question. ' \
                     f'3) The keywords can be supplements or extensions to the original question, not limited to the original question.' \
                     f'Question below:{item}'


    key_word=get_keywords(keyword_prompt)
    key_words.append(key_word)
    print(key_word)


for word in key_words:
    url=get_url(word)

    #保存论文爬虫结果至csv文件中
    file_path="./original_arxiv.csv"
    start_page = 0
    end_page = 4
    scrape_arxiv(start_page,end_page,url,file_path)


#消除重复的论文
new_file_path="./new_arxiv.csv"
df = pd.read_csv(file_path)
# 删除重复行，保留第一次出现的行
df_unique = df.drop_duplicates(subset='ArXiv Number', keep='first')
# 将清理后的数据保存到新的 CSV 文件
df_unique.to_csv(new_file_path, index=False)


new_file_path="./new_arxiv.csv"
#获得每篇文章的引用次数
citation_file_path="./citation_arxiv.csv"
df_1 = pd.read_csv(citation_file_path)

start_index=0
#end_index=
for index, row in df_1.iloc[start_index:128].iterrows():
    # 应用处理函数到 'ArXiv Number' 列
    citation_number = get_citation_Number(row['ArXiv Number'])
    # 更新 'Citation Number' 列

    df_1.at[index, 'Citation Number'] = citation_number
    # 将更新写回文件，这样即使过程中断，已处理的数据也不会丢失
    df_1.to_csv("./citation_arxiv.csv", index=False)

"""









