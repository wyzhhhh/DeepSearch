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
from langchain.document_transformers import EmbeddingsRedundantFilter



pdf_file_name = f"/mnt/mydisk/wangyz/Research_agent/pdf_download/{uuid.uuid4()}.pdf"

client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
)

#ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=04573ee80371d959df012c504261d2d9&Split=&Address=&isp=&poolnumber=0&qty=1"
ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=49aaa3d6f86cfdd65477f2d387677f37&Split=&Address=&isp=&poolnumber=0&qty=1"

total_start_time = time.time()
#question = "How do genetic variations in the APOE gene influence an individual's susceptibility to late-onset Alzheimer's disease, and what role do these variations play in the physiological mechanisms of the disease?"


block1_start_time = time.time()

question = input("请输入你的问题：")
#question = "How do genetic variations in the APOE gene influence an individual's susceptibility to late-onset Alzheimer's disease, and what role do these variations play in the physiological mechanisms of the disease?"
keyword_prompt = f'Please extract 3 most relevant keywords based on the following question,' \
         f'You must respond with a list of strings in the following format: ["keyword 1", "keyword 2", "keyword 3"].' \
         f'Requirement: 1) The keywords should be used to search for relevant academic papers. ' \
         f'2) The keywords should summarize the main idea of the question. ' \
         f'3) The keywords can be supplements or extensions to the original question, not limited to the original question.' \
         f'Question below:{question}' 

keywords_content = get_keywords(keyword_prompt)
print("key_words:", keywords_content)



#pubmed用例
pubmed_url = create_pubmed_url(keywords_content,type="abstract") #生成爬虫的url
print("开始爬取候选论文....")
#保存论文爬虫结果至csv文件中
start_page = 1
end_page = 5
ori_file_path="/mnt/mydisk/wangyz/Research_agent/csv_download/arxivpa_original_result.csv"
df = scrape_format_abstract(start_page=start_page, end_page=end_page, base_url=pubmed_url, file_path=ori_file_path)
# 删除重复行，保留第一次出现的行
df_unique = df.drop_duplicates(subset='PMID', keep='first')
# 将清理后的数据保存到新的 CSV 文件
new_file_path="/mnt/mydisk/wangyz/Research_agent/csv_download/arxivpa_afterFusion_result.csv"
df_unique.to_csv(new_file_path, index=False)
top_docs_df = df_unique
block1_end_time = time.time()
print("爬取结束....")
print("第一个爬虫得到候选论文列表花费时间：",block1_end_time-block1_start_time)


print("根据被引数量和相似度计算分数....")
#new_file_path="/mnt/mydisk/wangyz/Research_agent/csv_download/arxivpa_afterFusion_result.csv"
block2_start_time = time.time()
top_docs_df = pd.read_csv(new_file_path,encoding='ISO-8859-1')

# 根据被引数量进行排名
rerank_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/rerank_jina.csv"
afterrerank_df = Jina_rerank(top_docs_df,question,300)
#afterrerank_df = local_rerank(top_docs_df,question,300)
afternorma_df = Normalization(afterrerank_df,type = "Cited Number")#将被引数量正则化
weight_percentile = 4
weight_relevance = 6

# 确保'Percentile_Normalized'和'relevance_score'列是数值类型
afternorma_df['cited_Normalized'] = pd.to_numeric(afternorma_df['cited_Normalized'], errors='coerce')
afternorma_df['relevance_score'] = pd.to_numeric(afternorma_df['relevance_score'], errors='coerce')

# 计算final_score
afternorma_df['final_score'] = (afternorma_df['cited_Normalized'] * weight_percentile + 
                     afternorma_df['relevance_score'] * weight_relevance) / (weight_percentile + weight_relevance)
afternorma_df.to_csv(rerank_path, index=False)
block2_end_time = time.time()
print("得到核心论文中....")
print("总共耗时：",block2_end_time-block2_start_time)







#寻找最终分数最高的那一篇文章作为core paper
sorted_df = afternorma_df.sort_values(by='final_score', ascending=False).reset_index()
# 循环直到找到PMID, PMCID, DOI都不为空的行
for index, row in sorted_df.iterrows():
    if row['PMID'] is not None and row['PMCID'] is not None and row['DOI'] is not None:
        # 提取所需信息
        max_pmcid = row['PMCID']
        max_pmid = row['PMID']
        max_doi = row['DOI']
        max_citation_number = row['Cited Number']
        break
else:
    # 如果没有找到满足条件的行，则打印提示信息
    print("No row found with non-empty PMID, PMCID, and DOI.")

max_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{max_pmcid}/pdf/" # 核心论文的pdf链接
# 设置核心论文的references及citations的保存路径
references_filename =f"/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmcid}_references.csv"
citations_filename = f"/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmcid}_citations.csv"
print("被引量最多的文章pmcid:", max_pmcid)
print("被引量最多的文章pmid:", max_pmid)
print("被引量最多的文章doi:", max_doi)
print("被引量为：", max_citation_number)
# 得到核心论文的相关信息
title,abstract=extract_pdf_url(max_pmid)
core_paper={"title":title,"abstract":abstract}




# 得到核心论文的references以及citations
print("开始爬取核心论文的references以及citations....")
block3_start_time = time.time()
session = requests.Session()
get_references_citations(session,citations_filename, references_filename,doi=max_doi)#semantic sholoar api爬取是否重要等相关信息
#pubmed api爬取摘要等相关信息
first_page_url = f"https://pubmed.ncbi.nlm.nih.gov/?size=200&linkname=pubmed_pubmed_citedin&from_uid={max_pmid}"
total_pages = get_total_pages(first_page_url)
base_url="https://pubmed.ncbi.nlm.nih.gov/"
citedby_path = f"/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmid}_citedby.csv"
with open(citedby_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["PMID", "Title", "DOI", "Abstract", "PMCID", "Journal_title","Journal_id"])
scrape_format_pubmed(1, total_pages, base_url, citedby_path, pmID=max_pmid)

block3_end_time = time.time()
print("核心论文的references以及citations爬取完毕....")
print("该过程总共花费的时间为：",block3_end_time-block3_start_time)





block4_start_time = time.time()
similarity_output_path = f"/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmcid}_similarity_output.csv"
# 通过IP池得到多个ip
proxies = []
max_workers = 20 
for i in range(max_workers):
    response = requests.get(ip_api_url)
    ip = response.text.strip()  # 去除末尾的多余字符
    proxies.append({"HTTP": f"HTTP://{ip}", "HTTPS": f"HTTP://{ip}"})
    print(f"添加代理: {ip}")

# 设置URL和请求头
headers = {'Connection': 'close' }
session = requests.Session()


#如果两个文件都存在的话则合并做后续处理
if os.path.exists(citations_filename):
    # 得到核心论文的references列表
    df_origi = pd.read_csv(references_filename)
    origi_references_list = df_origi['Semantic Scholar ID'].tolist()
    # 将两个api得到的citations信息合并
    df1 = pd.read_csv(citations_filename)
    df2 = pd.read_csv(citedby_path)
    #清楚为空的DOI号
    df1 = df1.dropna(subset=['DOI'])
    df2 = df2.dropna(subset=['DOI'])
    df2['Title'] = df2['Title'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))
    df2['Abstract'] = df2['Abstract'].apply(lambda x: x.replace('\n', ' ').replace('\r', '') if isinstance(x, str) else None)
    # 找出所有非字符串值的索引
    non_string_indices = df2[df2['Abstract'].isna()].index
    # 打印出这些索引和对应的原始值
    for index in non_string_indices:
        print(f"Non-string value found at index {index}: {df2.loc[index, 'Abstract']}")
    #根据DOI号合并
    merged_df = pd.merge(df2, df1, on='DOI', how='left', indicator=True)
    merged_df['Title'] = merged_df['Title_x'].combine_first(merged_df['Title_y'])
    merged_df.drop(['Title_x', 'Title_y'], axis=1, inplace=True)
    columns_to_keep = ['DOI', 'Title', 'Year', 'Abstract', 'Semantic Scholar ID', 'URL', 'PMID', 'PMCID', 'Intent', 'isInfluential', 'Journal_title', 'Journal_id']
    kept_df = merged_df[merged_df['_merge'] == 'both'][columns_to_keep]

    rename_dict = {
        'Year': 'year',
        'Semantic Scholar ID': 'Semantic Scholar ID',
        'URL': 'url',
        'Intent': 'intent',
        'isInfluential': 'isinfluential',
        'Journal_title': 'journal_title',
        'Journal_ID': 'journal_id'
    }
    kept_df.rename(columns=rename_dict, inplace=True)
    kept_df.replace({np.nan: None}, inplace=True)
    kept_df.to_csv('/mnt/mydisk/wangyz/Research_agent/csv_download/merged_citations.csv', index=False, na_rep='')
    rerank_citation_df = Jina_rerank(kept_df,question,200) #将合并后的citation进行rerank，计算relevance

        

    # 计算references similarity
    
    futures = {}
    similarity_results = {}
    impact_factor_results = {}
    cited_number_results = {}

    # 创建两个线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor1, \
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor2:

        # 提交 get_similarity 任务
        for i, paperid in enumerate(rerank_citation_df['Semantic Scholar ID']):
            proxy = proxies[i % len(proxies)]  # 通过 Semantic Scholar ID 对代理列表进行轮询
            future = executor1.submit(get_similarity, session, paperid, origi_references_list, proxy)
            futures[future] = paperid  # 存储 future 和 paperid 的映射

        # 提交 impact_factor 任务
        for journal_id in rerank_citation_df['Journal_id'].unique():  # 确保对每个 journal_id 只计算一次
            future = executor2.submit(impact_factor, str(journal_id))
            futures[future] = journal_id

        # 收集 get_similarity 的结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:  # 确保返回的结果不是 None
                    if futures[future] in rerank_citation_df['Semantic Scholar ID'].values:
                        similarity_results[futures[future]] = result[1]  # 假设 result 是一个元组 (similarity, cited_number)
                        cited_number_results[futures[future]] = result[2]
                    elif futures[future] in rerank_citation_df['Journal_id'].values:
                        impact_factor_results[futures[future]] = result
            except Exception as e:
                print(f"Error: {e}")

    # 更新 rerank_citation_df DataFrame，添加 similarity 和 impact_factor 列
    if similarity_results:
        rerank_citation_df['similarity'] = rerank_citation_df['Semantic Scholar ID'].map(similarity_results)
    if cited_number_results:
        rerank_citation_df['Cited Number'] = rerank_citation_df['Semantic Scholar ID'].map(cited_number_results)
    if impact_factor_results:
        rerank_citation_df['impact_factor'] = rerank_citation_df['Journal_id'].map(impact_factor_results)

    #挑选出最相关且重要的几篇文章
    similarity_df = Normalization(rerank_citation_df,type="Cited Number")
    similarity_df = Normalization(similarity_df,type="impact_factor")
    final_score = (
        0.3 * similarity_df['relevance_score'] +
        0.4 * similarity_df['similarity'] +
        0.15 * similarity_df['cited_Normalized'] +
        0.15 * similarity_df['impact_Normalized']
    )





else:
#semantic scholar api 没有收录被引信息的话只用citedby一个文件处理

    df2 = pd.read_csv(citedby_path)
    df2 = df2.dropna(subset=['DOI'])
    df2['Title'] = df2['Title'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))
    df2['Abstract'] = df2['Abstract'].apply(lambda x: x.replace('\n', ' ').replace('\r', '') if isinstance(x, str) else None)

    rerank_citation_df = Jina_rerank(df2,question,200)
    futures = {}
    impact_factor_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(impact_factor, str(journal_id)): journal_id for journal_id in rerank_citation_df['Journal_id'].unique()}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    if futures[future] in rerank_citation_df['Journal_id'].values:
                        impact_factor_results[futures[future]] = result
            except Exception as e:
                print(f"Error: {e}")

    if impact_factor_results:
        rerank_citation_df['impact_factor'] = rerank_citation_df['Journal_id'].map(impact_factor_results)



    similarity_df = Normalization(rerank_citation_df,type="impact_factor")
    final_score = (
        0.65 * similarity_df['relevance_score'] +
        0.35 * similarity_df['impact_Normalized']
    )


# 将计算结果作为一个新列添加到 DataFrame 中
rerank_citation_df['final_score'] = final_score
rerank_citation_df.to_csv(similarity_output_path,index=False)
# 根据相似度确定最终的pdf链接
pdf_urls, abstract_list, title_list,relatedpapers = extract_pdf_similarity(similarity_output_path,type="pubmed",top_n=4)
pdf_urls.append(max_url)
print("The last PDF URLs found:")
print(pdf_urls)


pdfs = []
for pdf_url in pdf_urls:
    parts = pdf_url.split('/')
    pmc_id = parts[5]
    pdf_save_path = f"/mnt/mydisk/wangyz/Research_agent/top_document/{pmc_id}.pdf"
    download_pdf(pdf_url, pdf_save_path)  # 下载pdf文件
    pdfs.append(pdf_save_path)

block4_end_time = time.time()
print("Both similarity and impact factor calculations are complete.")
print("Time taken for this process:", block4_end_time - block4_start_time)



block5_start_time = time.time()

voyage_embeddings = VoyageAIEmbeddings(voyage_api_key="pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE", model="voyage-large-2")
jina_embeddings = JinaEmbeddings(jina_api_key="jina_e61db1cedc3c4b3fafe4e7632d8e00c5VW6PpZV4Y0wRoA4yKayeoW3JQcod", model_name="jina-embeddings-v2-base-en")

# Prompts dictionary
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

combined_text = []
recommender = SemanticSearch()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {executor.submit(process_document, url, prompts,recommender): url for url in pdfs}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            combined_text.append(result)
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")


for i in range(len(combined_text)):
    combined_text[i]['url'] = pdf_urls[i]
"""
#使用langchain读pdf文件


combined_text = defaultdict(dict)

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
redundant_filter = EmbeddingsRedundantFilter(embeddings=voyage_embeddings)
relevant_filter = EmbeddingsFilter(embeddings=voyage_embeddings, similarity_threshold=0.38)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_pdf_url, index, pdf_url,voyage_embeddings,pipeline_compressor,prompts): index for index, pdf_url in enumerate(pdfs)}

    for future in as_completed(futures):
        index, local_context = future.result()
        combined_text[index] = local_context



"""
report_content = {}
for prompt_key, prompt_value in prompts.items():
    report = generate_report(context=combined_text, question=question, prompt_type=prompt_key, paper=core_paper, relatedPaper=relatedpapers)
    report_content[prompt_key] = report
block5_end_time = time.time()
print("内容读取完毕...")
print("这部分花费时间为：",block5_end_time-block5_start_time)






print("开始获得新idea....")
block6_start_time = time.time()
# 获取新idea的过程

initial_response = generate_report(combined_text, question, prompt_type="idea", paper=core_paper,
                                   relatedPaper=relatedpapers)
report_content["idea"] = initial_response
problem, rationale = extract_content(initial_response, "Problem:")

second_response = generate_report(combined_text, question, prompt_type="method", paper=core_paper,
                                  relatedPaper=relatedpapers,
                                  researchProblem=problem, researchProblemRationale=rationale)
report_content["method"] = second_response
method, method_rationale = extract_content(second_response, "Method:")

final_response = generate_report(combined_text, question, prompt_type="experiment", paper=core_paper,
                                 relatedPaper=relatedpapers,
                                 researchProblem=problem, researchProblemRationale=rationale,
                                 scientificMethod=method, scientificMethodRationale=method_rationale)
report_content["experiment"] = final_response

print("--------------------new idea reports--------------------------")
print(report_content)

block6_end_time = time.time()
print("新idea已获得....")
print("这部分花费时间为：",block6_end_time-block6_start_time)


# 根据每部分内容生成总结
block7_start_time = time.time()
print("生成总结中....")



#添加相关文章信息


DOIs, abstracts, titles, IFs = extract_more(similarity_output_path,top_n=10)
more_content = ""
for i in range(10):
    more_content = more_content +f"Paper {i+1}"+"\n"+"Title:"+titles[i]+"\n\n"+"Abstract:"+abstracts[i]+"\n"+"DOI:"+DOIs[i]+"\n"+"The impact factor:"+str(IFs[i])+"\n\n\n\n"
report_content["More related paper"] = more_content


print(report_content)

raw_text = dict_to_text(report_content)
content = preprocess_text(raw_text)
create_pdf(content, pdf_file_name)

block7_end_time = time.time()
print("花费时间为：",block7_end_time-block7_start_time)
print(f"文件已保存为：{pdf_file_name}")

print("总花销时间为：",block7_end_time-total_start_time)
print("第一部分花销：",block1_start_time-block1_end_time)
print("第二部分花销：",block2_start_time-block2_end_time)
print("第三部分花销：",block3_start_time-block3_end_time)
print("第四部分花销：",block4_start_time-block4_end_time)
print("第五部分花销：",block5_start_time-block5_end_time)
print("第六部分花销：",block6_start_time-block6_end_time)
print("第七部分花销：",block7_start_time-block7_end_time)



