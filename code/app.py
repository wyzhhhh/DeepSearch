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
from langchain.document_transformers import EmbeddingsRedundantFilter
from RSR_score import *

pdf_file_name = f"/mnt/mydisk/wangyz/Research_agent/pdf_download/{uuid.uuid4()}.pdf"
client = OpenAI(
    #base_url="https://kapkey.chatgptapi.org.cn/v1",
    #api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"


    base_url = "https://api.xiaoai.plus/v1",
    api_key = "sk-hz4C02ZEZUjbkk0aE92028468246454793Bc6649F0Bb1b9e"

    #base_url = "https://api.closeai-proxy.xyz/v1",
    #api_key = "sk-v1Y3L4qrAPFGKMIAJ4wZ5H8eJxAuH97GYnA4iFm0pwqlKaJx"
)

#ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=04573ee80371d959df012c504261d2d9&Split=&Address=&isp=&poolnumber=0&qty=1"
ip_api_url = "http://zhuoyuekeji.zhuoyuejituan.com:66/SML.aspx?action=GetIP&OrderNumber=49aaa3d6f86cfdd65477f2d387677f37&Split=&Address=&isp=&poolnumber=0&qty=1"
initial_jina_api_key = "jina_a6f1d31fb5b34b3fa5ddfeff38105879lhkP5FSfnhZBJRNp0qFUwoS-XX5G"
voyage_api_key = "pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE"


# 设置页面标题
st.title('Research Agent')
question = st.text_input("Please enter your question:")
#OPENAI_KEY = st.text_input("Please enter your openai key:")
jina_api_key = st.text_input("Please enter your jina reranker key:", value=initial_jina_api_key)
if st.button('Get Started'):
    if question:

        #根据问题得到关键词并输出

        total_start_time = time.time()

        block1_start_time = time.time()
        keyword_prompt = f'Please extract 3 most relevant keywords based on the following question,' \
                f'You must respond with a list of strings in the following format: ["keyword 1", "keyword 2", "keyword 3"].' \
                f'Requirement: 1) The keywords should be used to search for relevant academic papers. ' \
                f'2) The keywords should summarize the main idea of the question. ' \
                f'3) The keywords can be supplements or extensions to the original question, not limited to the original question.' \
                f'Question below:{question}' 

        keywords_content = get_keywords(keyword_prompt)
        print("key_words:", keywords_content)
        st.write("🔜The key words are as follows:", keywords_content)


        abstract_url = create_pubmed_url(keywords_content,type="abstract")
        pubmed_url = create_pubmed_url(keywords_content,type="pubmed")

        st.write(f"🔎 Running research for '{question}'...")


        st.info('🕗Starting the scraping process...')
        start_page = 1
        end_page = 5
        ori_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/abstract_ori_result.csv"
        pubmed_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/pubmed_ori_result.csv"
        # 创建并行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_abstract = executor.submit(scrape_format_abstract, start_page, end_page, abstract_url, ori_file_path)
            future_pubmed = executor.submit(scrape_format_pubmed_url, start_page, end_page, pubmed_url, pubmed_file_path)
            try:
                df1 = future_abstract.result()
            except Exception as e:
                print(f"Error occurred while fetching abstract data: {e}")
                df1 = None

            try:
                df2 = future_pubmed.result()
            except Exception as e:
                print(f"Error occurred while fetching PubMed data: {e}")
                df2 = None 


        # 处理abstract数据集，去重并保存
        df_unique = df1.drop_duplicates(subset='PMID', keep='first')
        # 合并两个DataFrame
        merged_df = pd.merge(df_unique, df2, on='PMID', how='inner')
        
        block1_end_time = time.time()
        st.write("😊Finalized curating a list of relevant papers!")


        futures = {}
        impact_factor_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor1:
                # 提交 impact_factor 任务
                for journal_id in merged_df['Journal_id'].unique():  # 确保对每个 journal_id 只计算一次
                    future = executor1.submit(impact_factor, str(journal_id))
                    futures[future] = journal_id

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:  # 确保返回的结果不是 None
                            if futures[future] in merged_df['Journal_id'].values:
                                impact_factor_results[futures[future]] = result
                    except Exception as e:
                        print(f"Error: {e}")

        if impact_factor_results:
                merged_df['impact_factor'] = merged_df['Journal_id'].map(impact_factor_results)

        merged_file_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/merged_afterfusion.csv"
        merged_df.to_csv(merged_file_path, index=False)
        st.success('✅Scraping process completed successfully.')


        st.info("🧠Calculating correlations and obtaining core papers...")
        #st.write("根据被引数量和相似度计算分数....")
        block2_start_time = time.time()
        top_docs_df = pd.read_csv(merged_file_path,encoding='ISO-8859-1')
        
        # 根据被引数量进行排名
        #rerank_path = "/mnt/mydisk/wangyz/Research_agent/csv_download/rerank_jina.csv"
        afterrerank_df = Jina_rerank(top_docs_df,question,300,jina_api_key)
        #afterrerank_df = local_rerank(top_docs_df,question,300)
        afternorma_df = Normalization(afterrerank_df,type = "Cited Number")#将被引数量正则化
        afternorma_df = Normalization(afternorma_df,type = "impact_factor")
        weight_percentile = 2
        weight_relevance = 6
        weight_impact = 2

        # 确保'Percentile_Normalized'和'relevance_score'列是数值类型
        afternorma_df['cited_Normalized'] = pd.to_numeric(afternorma_df['cited_Normalized'], errors='coerce')
        afternorma_df['relevance_score'] = pd.to_numeric(afternorma_df['relevance_score'], errors='coerce')
        afternorma_df['impact_factor'] = pd.to_numeric(afternorma_df['impact_factor'], errors='coerce')

        # 计算final_score
        afternorma_df['final_score'] = (afternorma_df['cited_Normalized'] * weight_percentile + 
                            afternorma_df['relevance_score'] * weight_relevance +
                            afternorma_df['impact_Normalized'] * weight_impact)/ (weight_percentile + weight_relevance + weight_impact)
        #afternorma_df.to_csv(rerank_path, index=False)


        #RSR方法计算分数
        selected_data = afternorma_df[['Cited Number', 'relevance_score', 'impact_factor']]
        core_rsr_path = f'/mnt/mydisk/wangyz/Research_agent/RSR_core/{uuid.uuid4()}_core_rsr'
        RSR_result, distribution = rsrAnalysis(selected_data, file_name=core_rsr_path)
        print("The list of core papers are saved in ",core_rsr_path)
        RSR_result = RSR_result.sort_index()
        RSR_result = RSR_result.reindex(afternorma_df.index)
        afternorma_df['RSR'] = RSR_result['RSR']
        afternorma_df['Probit'] = RSR_result['Probit']
        
        

        block2_end_time = time.time()
        










        #寻找最终分数最高的那一篇文章作为core paper
        sorted_df = afternorma_df.sort_values(by='Probit', ascending=False).reset_index()
        # 循环直到找到PMID, PMCID, DOI都不为空的行

        found = False
        for index, row in sorted_df.iterrows():
            # 确保所有列均非空
            if pd.notna(row['PMID']) and pd.notna(row['PMCID']) and pd.notna(row['DOI']):
                max_pmcid = row['PMCID']
                max_pmid = row['PMID']
                max_doi = row['DOI']
                max_citation_number = row['Cited Number']
                found = True
                break

        if not found:
            print("No row found with non-empty PMID, PMCID, and DOI.")


        rerank_path = f"/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmid}_core_rerank_jina.csv"
        afternorma_df.to_csv(rerank_path, index=False)

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
        st.write("📜Title of the core paper:",title)
        st.write(f"📜Abstract of the core paepr:",abstract)
        st.write(f"📜PMCID of the core paper:",max_pmcid)
        st.write(f"📜PMID  of the core paper:",max_pmid)
        st.write(f"📜DOI   of the core paper:",max_doi)



        # 得到核心论文的references以及citations
        print("🔎 Running search for references and citations of the core paper...")
        st.info("🧠 Starting search for references and citations of the core paper...")
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

        st.info('🕗Starting the scraping process...')
        scrape_format_pubmed(1, total_pages, base_url, citedby_path, pmID=max_pmid)

        block3_end_time = time.time()
        print("核心论文的references以及citations爬取完毕....")
        print("该过程总共花费的时间为：",block3_end_time-block3_start_time)
        st.success('✅Scraping process completed successfully.')
        st.write(f"😊Finalized curating the references and citations of core paper!")







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
        if os.path.exists(citations_filename) and os.path.exists(references_filename):
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
            rerank_citation_df = Jina_rerank(kept_df,question,200,jina_api_key) #将合并后的citation进行rerank，计算relevance

                

            # 计算references similarity
            
            futures = {}
            similarity_results = {}
            impact_factor_results = {}
            cited_number_results = {}

            total_tasks = len(rerank_citation_df['Semantic Scholar ID'].unique())+len(rerank_citation_df['Journal_id'].unique())
            task_count = 0
            progress_bar_simi = st.progress(0)
            st.info("🕗Starting the calculating process...")


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
                    task_count += 1
                    progress_bar_simi.progress(task_count / total_tasks)
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
            
            st.success('✅Calculating process completed successfully.')

            # 更新 rerank_citation_df DataFrame，添加 similarity 和 impact_factor 列
            if similarity_results:
                rerank_citation_df['similarity'] = rerank_citation_df['Semantic Scholar ID'].map(similarity_results)
            if cited_number_results:
                rerank_citation_df['Cited Number'] = rerank_citation_df['Semantic Scholar ID'].map(cited_number_results)
            if impact_factor_results:
                rerank_citation_df['impact_factor'] = rerank_citation_df['Journal_id'].map(impact_factor_results)




            
            #RSR方法计算分数
            selected_data = rerank_citation_df[['Cited Number', 'relevance_score', 'impact_factor','similarity']]
            RSR_result, distribution = rsrAnalysis(selected_data, file_name=f'/mnt/mydisk/wangyz/Research_agent/RSR_core/{max_pmid}_rsr')
            RSR_result = RSR_result.sort_index()
            RSR_result = RSR_result.reindex(rerank_citation_df.index)
            rerank_citation_df['RSR'] = RSR_result['RSR']
            rerank_citation_df['Probit'] = RSR_result['Probit']
            rerank_citation_df.to_csv(f'/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmid}_updated_data.csv', index=False)
            



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

            rerank_citation_df = Jina_rerank(df2,question,200,jina_api_key)
            futures = {}
            impact_factor_results = {}

            total_tasks = len(rerank_citation_df['Journal_id'].unique())
            task_count = 0
            progress_bar_if = st.progress(0)
            st.info('🕗Starting the calculating process...')

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(impact_factor, str(journal_id)): journal_id for journal_id in rerank_citation_df['Journal_id'].unique()}
                
                for future in concurrent.futures.as_completed(futures):
                    task_count += 1
                    progress_bar_if.progress(task_count / total_tasks)
                    try:
                        result = future.result()
                        if result is not None:
                            if futures[future] in rerank_citation_df['Journal_id'].values:
                                impact_factor_results[futures[future]] = result
                    except Exception as e:
                        print(f"Error: {e}")
            st.success('✅Calculating process completed successfully.')

            if impact_factor_results:
                rerank_citation_df['impact_factor'] = rerank_citation_df['Journal_id'].map(impact_factor_results)


            
            #RSR计算分数
            selected_data = rerank_citation_df[['relevance_score', 'impact_factor']]
            RSR_result, RSR_distribution = rsrAnalysis(selected_data, file_name=f'/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmid}_rsr')
            RSR_result = RSR_result.sort_index()
            RSR_result = RSR_result.reindex(rerank_citation_df.index)
            rerank_citation_df['RSR'] = RSR_result['RSR']
            rerank_citation_df['Probit'] = RSR_result['Probit']
            rerank_citation_df.to_csv(f'/mnt/mydisk/wangyz/Research_agent/csv_download/{max_pmid}_updated_data.csv', index=False)
            


            similarity_df = Normalization(rerank_citation_df,type="impact_factor")
            final_score = (
                0.65 * similarity_df['relevance_score'] +
                0.35 * similarity_df['impact_Normalized']
            )
        


        # 将计算结果作为一个新列添加到 DataFrame 中
        rerank_citation_df['final_score'] = final_score
        rerank_citation_df['RSR'] = RSR_result['RSR']
        rerank_citation_df.to_csv(similarity_output_path,index=False)
        # 根据相似度确定最终的pdf链接
        pdf_urls, abstract_list, title_list,relatedpapers = extract_pdf_similarity(similarity_output_path,type="pubmed",top_n=4)
        pdf_urls.append(max_url)
        print("The last PDF URLs found:")
        print(pdf_urls)
        st.write(f"🔜The final paper urls as follows:",pdf_urls)

        pdfs = []
        total_pdfs = len(pdf_urls)
        progress_bar_pdf = st.progress(0)
        for i,pdf_url in enumerate(pdf_urls):
            parts = pdf_url.split('/')
            pmc_id = parts[5]
            pdf_save_path = f"/mnt/mydisk/wangyz/Research_agent/top_document/{pmc_id}.pdf"
            download_pdf(pdf_url, pdf_save_path)  # 下载pdf文件
            overall_progress = (i + 1) / total_pdfs
            progress_bar_pdf.progress(overall_progress)  # 更新进度条
            pdfs.append(pdf_save_path)
        st.success("✅All pdf files have been downloaded!")
        block4_end_time = time.time()
        print("Both similarity and impact factor calculations are complete.")
        print("Time taken for this process:", block4_end_time - block4_start_time)
        




                            
        st.info("🧠Starting generate the content of final paper...")
        block5_start_time = time.time()
        voyage_embeddings = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model="voyage-large-2")
        jina_embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en")

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
        progress_bar_retrie = st.progress(0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pdfs)-1) as executor:
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
                    #st.error(f"{url} generated an exception: {exc}")
                completed += 1
                progress_bar_retrie.progress(completed / total_pdfs)


        for i in range(len(combined_text)):
            combined_text[i]['url'] = pdf_urls[i]


        report_content = {}
        progress_bar_report = st.progress(0)
        report_item = 0
        for prompt_key, prompt_value in prompts.items():
            report = generate_report(context=combined_text, question=question, prompt_type=prompt_key, paper=core_paper, relatedPaper=relatedpapers)
            report_content[prompt_key] = report
            report_item += 1
            progress_bar_report.progress(report_item / 3)
        block5_end_time = time.time()

        print("内容读取完毕...")
        print("这部分花费时间为：",block5_end_time-block5_start_time)

        st.success("✅The content of the final report already done!")




        st.info("🧠Starting Generate the new idea and experiment...")
        print("开始获得新idea....")
        block6_start_time = time.time()
        # 获取新idea的过程
        with st.spinner("⏳ Generating new ideas..."):
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
        st.success("✅The new idea have been generated.")

        print("--------------------new idea reports--------------------------")
        print(report_content)

        block6_end_time = time.time()
        print("新idea已获得....")
        print("这部分花费时间为：",block6_end_time-block6_start_time)


        # 根据每部分内容生成总结
        block7_start_time = time.time()
        print("生成总结中....")

        
        DOIs, abstracts, titles, IFs = extract_more(similarity_output_path,top_n=10)
        more_content = ""
        num_papers = len(DOIs)
        if num_papers >= 10:
            num_papers = 10
        for i in range(num_papers):

            more_content += (f"Paper {i+1}\n"
                            f"Title: {titles[i]}\n\n"
                            f"Abstract: {abstracts[i]}\n\n"
                            f"DOI: {DOIs[i]}\n"
                            f"The impact factor: {IFs[i]}\n\n\n\n")
        
        report_content["More related paper"] = more_content


        report_sections = ["Introduction","Literature","Discussion","idea","method","experiment","More related paper"]
        for section in report_sections:
            if section in report_content:
                st.header(section.capitalize())
                st.markdown(report_content[section], unsafe_allow_html=True) 


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





    else:
        st.text("The input is None!")


