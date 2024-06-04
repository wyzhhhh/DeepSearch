import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re
import pandas as pd
import concurrent.futures


def scrape_arxiv(start_page, end_page,base_url,file_path):
    results = []

    for page in range(start_page, end_page + 1):

        url = f"{base_url}&start={page * 200}"
        response = requests.get(url)
        while response.status_code == 403:
            time.sleep(500 + random.uniform(0, 500))
            response = requests.get(url)
            print(response.status_code)
        if response.status_code == 200:
            print("Successfully access " + url + "!")

        soup = BeautifulSoup(response.text, 'html.parser')

        for item in soup.find_all('li', class_='arxiv-result'):
            title = item.find('p', class_='title').text.strip()

            abstract_span = item.find('span', class_='abstract-full')
            if abstract_span:
                # Remove the "Less" link
                last_link = abstract_span.find('a')
                if last_link:
                    last_link.decompose()
                abstract = abstract_span.text.strip()
            else:
                abstract = item.find('span', class_='abstract-short').text.strip()

            arxiv_number = item.find('a').text.strip()
            arxiv_number = arxiv_number.split(':')[1]

            results.append([arxiv_number, title, abstract])

    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ArXiv Number', 'Title', 'Abstract'])
        writer.writerows(results)



def scrape_pubmed(start_page, end_page, base_url, file_path):

    results = []

    for page in range(start_page, end_page + 1):

        url = f"{base_url}&page={page}"
        response = requests.get(url)
        while response.status_code == 403:
            time.sleep(500 + random.uniform(0, 500))
            response = requests.get(url)
            print(response.status_code)
        if response.status_code == 200:
            print("Successfully access " + url + "!")

        soup = BeautifulSoup(response.text, 'html.parser')


        # 查找所有包含文章标题和DOI的容器
        articles = soup.find_all('div', class_='docsum-wrap')

        # 遍历每个文章容器，提取标题和完整DOI号
        for article in articles:
            # 提取标题
            title_tag = article.find('a', class_='docsum-title')
            title = title_tag.text.strip() if title_tag else "标题未找到"

            doi_tag = article.find('span', class_='docsum-journal-citation', string=lambda text: 'doi' in text)
            if doi_tag:
                doi_match = re.search(r'doi: (\S+)', doi_tag.text)  # 使用正则表达式匹配，提取'.'之前的所有内容，并确保不是最后一个字符
                doi = doi_match.group(1) if doi_match else "DOI格式不正确"
                doi = doi[:-1]
            else:
                doi = "DOI未找到"

            print(f"标题: {title}\nDOI号: {doi}\n")

            results.append([title,doi])



    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'DOI'])
        writer.writerows(results)










def get_total_pages(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    total_pages_label = soup.find('label', class_='of-total-pages')
    if total_pages_label:
        total_pages = int(total_pages_label.text.split()[-1])
        print(f"Total pages: {total_pages}")
    else:
        print("Total pages label not found.")
    return total_pages

def scrape_format_pubmed_url(start_page, end_page, base_url, file_path):
    
    data = []  # List to hold the data dictionaries

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for page in range(start_page, end_page + 1):
        url = f"{base_url}&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pre_tag = soup.find('pre', class_='search-results-chunk')
            if pre_tag:
                articles = pre_tag.text.strip().split('PMID-')
                for article in articles:
                    if article:
                        if not article.startswith('PMID-'):
                            article = 'PMID-' + article  # Ensure the split item starts correctly

                        pmid = get_field(article, 'PMID-', ['\n'])
                        jt = get_field(article, 'JT  -', ['\n'])
                        jid = get_field(article, 'JID -', ['\n'])
                        if jid.startswith('0'):
                            jid = jid.lstrip('0')
                        if pmid and jt and jid:  # Ensure all fields are non-empty
                            data.append({'PMID': pmid, 'journal_title': jt, 'Journal_id': jid})
        else:
            print(f"Failed to retrieve data from {url}: Status code {response.status_code}")

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df




def scrape_format_pubmed_page(page, start_page, end_page, base_url, file_path, proxy,pmID=None):
    # 确保只处理当前页码范围内的页面
    if start_page <= page <= end_page:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for current_page in range(page, page + 1):  # 只处理当前页
                if pmID:
                    url = f'{base_url}?page={current_page}&format=pubmed&size=200&linkname=pubmed_pubmed_citedin&from_uid={pmID}'
                else:
                    url = f'{base_url}&page={current_page}'
                print("URL:", url)
                response = requests.get(url, proxies=proxy,headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    pre_tag = soup.find('pre', class_='search-results-chunk')
                    if pre_tag:
                        articles = pre_tag.text.strip().split('PMID-')
                        for article in articles:
                            if article:
                                labels = ['LID -','CI  -','PG  -','AD  -', 'FAU -','AU  -']
                                #print("--------------------------------------------------")
                                #print(article)
                                if not article.startswith('PMID-'):
                                    article = 'PMID-' + article  # Ensure the split item starts correctly
                                
                                    pmid = get_field(article, 'PMID-', ['\n'])
                                    title = get_field(article, 'TI  -', labels)
                                    title = title.replace('\n', ' ')
                                    doi = get_field(article, 'AID -', ['\n'], must_include='[doi]')
                                    abstract = get_field(article, 'AB  -', labels)
                                    abstract = abstract.replace('\n', ' ')
                                    pmc = get_field(article, 'PMC -', ['\n'])
                                    jt = get_field(article, 'JT  -', ['\n'])
                                    jid = get_field(article, 'JID -', ['\n'])
                                    if jid.startswith('0'):
                                        jid = jid.lstrip('0')
                                    #print("-------------------------------------")
                                    #print(title)
                                writer.writerow([pmid, title, doi, abstract, pmc, jt, jid])
                else:
                    print(f"Failed to retrieve data from {url}: Status code {response.status_code}")

def scrape_format_pubmed(start_page, end_page, base_url, file_path, pmID=None):
    ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=49aaa3d6f86cfdd65477f2d387677f37&Split=&Address=&isp=&poolnumber=0&qty=1"
    proxies = []
    max_workers = end_page // 2 if end_page // 2 != 0 else 2
    for i in range(max_workers):
        response = requests.get(ip_api_url)
        ip = response.text.strip()  # 去除末尾的多余字符
        proxies.append({"HTTP": f"HTTP://{ip}", "HTTPS": f"HTTP://{ip}"})
        print(f"添加代理: {ip}")


    count = 0
    total_pages = end_page
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 使用5个线程
        future_to_page = {executor.submit(scrape_format_pubmed_page, page, 1, end_page, base_url, file_path, pmID=pmID, proxy=proxies[i % len(proxies)]): page for i, page in enumerate(range(1, end_page + 1))}
        for future in concurrent.futures.as_completed(future_to_page):
            page = future_to_page[future]
            try:
                future.result()  # 调用这个方法会阻塞直到爬取任务完成
                count += 1
                progress_bar.progress(count / total_pages)  # 更新进度条
            except Exception as exc:
                print(f'Page {page} generated an exception: {exc}')


def get_field(data, label, end_labels,must_include=None):
    start_index = 0
    valid_value = ''
    while True:
        start_index = data.find(label,start_index)
        if start_index == -1:
            return valid_value  # 如果字段不存在，返回空字符串
        start_index += len(label)
        
        # 查找所有可能的结束标签的最小索引
        min_end_index = len(data)
        for end_label in end_labels:
            end_index = data.find(end_label, start_index)
            if end_index != -1 and end_index < min_end_index:
                min_end_index = end_index

        extracted_value = data[start_index:min_end_index].strip()
        if must_include:
            if must_include in extracted_value:
                return extracted_value.split(' ')[0]  # 只返回匹配部分前的内容
        else:
            valid_value = ' '.join(extracted_value.split('\n')).strip()
        start_index = end_index







def scrape_page(page, base_url, proxy):
    # 初始化本页的存储列表
    titles = []
    abstracts = []
    pmids = []
    pmcids = []
    cited_numbers = []
    dois = []
    
    # 构造完整的URL
    url = f"{base_url}&page={page}"
    
    
    # 发送HTTP请求
    try:
        response = requests.get(url, proxies=proxy)
        if response.status_code == 200:
            # 解析网页
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all("div", class_="results-article")
            
            # 遍历每一个文章部分提取信息
            for article in articles:
                #print("---------------------------------------------")
                # 提取标题
                title = article.find("h1", {"class": "heading-title"})
                if title:
                    title_text = title.get_text(strip=True)
                else:
                    title_text = None
                #print("Title:",title_text)
                titles.append(title_text)

                # 提取摘要
                abstract_div = article.find('div', class_='abstract-content selected')
                if abstract_div:
                    abstract = abstract_div.get_text(strip=True)
                else:
                    abstract = None
                #print("Abstract:",abstract)
                abstracts.append(abstract)

                # 提取PMID
                pmid = article.find("strong", {"class": "current-id"})
                if pmid:
                    pmID = pmid.text.strip()
                else:
                    pmID = None
                #print("PMID:",pmID)
                pmids.append(pmID)

                # 提取PMCID
                pmc_span = article.find('span', class_='identifier pmc')
                if pmc_span and pmc_span.find('a'):
                    pmcid = pmc_span.find('a').text.strip()
                else:
                    pmcid = None
                #print("pmcid:",pmcid)
                pmcids.append(pmcid)

                # 提取DOI
                doi_span = article.find('span', class_='identifier doi')
                if doi_span and doi_span.find('a'):
                    doi = doi_span.find('a').text.strip()
                else:
                    doi = None
                #print("DOI:",doi)
                dois.append(doi)

                # 提取被引用数
                cited_number = 0
                citedby_text = article.find('li', class_='citedby-count')
                if citedby_text:
                    cited_number = re.search(r'\d+', citedby_text.text).group(0)
                #print("Cited Number:",cited_number)
                cited_numbers.append(cited_number)
    except Exception as e:
        print(f"Failed to retrieve page {page} using proxy {proxy}: {str(e)}")

    return {
        'Title': titles,
        'Abstract': abstracts,
        'PMID': pmids,
        'PMCID': pmcids,
        'DOI': dois,
        'Cited Number': cited_numbers
    }




def scrape_format_abstract(start_page, end_page, base_url, file_path):

    #设置ip代理池
    ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=49aaa3d6f86cfdd65477f2d387677f37&Split=&Address=&isp=&poolnumber=0&qty=1"
    proxies = []
    max_workers = 5
    for i in range(max_workers):
        response = requests.get(ip_api_url)
        ip = response.text.strip()  # 去除末尾的多余字符
        proxies.append({"HTTP": f"HTTP://{ip}", "HTTPS": f"HTTP://{ip}"})
        print(f"添加代理: {ip}")
    


    pages = list(range(start_page, end_page + 1))


    # 使用ThreadPoolExecutor并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(proxies)) as executor:
        futures = [executor.submit(scrape_page, page, base_url, proxies[i % len(proxies)]) for i, page in enumerate(pages)]
        #results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # 合并结果
    all_data = {
        'Title': [],
        'Abstract': [],
        'PMID': [],
        'PMCID': [],
        'DOI': [],
        'Cited Number': []
    }

    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        result = future.result()
        for key in all_data:
            all_data[key].extend(result[key])

    
    

    """
    for result in results:
        for key in all_data:
            all_data[key].extend(result[key])
    """
    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(all_data)
    df.to_csv(file_path, index=False)
    return df



