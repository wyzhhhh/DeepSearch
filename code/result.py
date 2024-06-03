from url_citation_number import *
from scrapearxiv import *
from semantic_api import *
from PDF_reader import *
from summarize import *
from save_pdf import *
import uuid
from pubmed import *
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import contextlib
import io

def handle(pmcid, doi):
    citation_path = "./output/csv_download/"
    reference_path = "./output/csv_download/"
    similarity_output = "./output/csv_download/"
    pdf_file_name = f"./output/pdf_download/{uuid.uuid4()}.pdf"

    client = OpenAI(
        base_url="https://kapkey.chatgptapi.org.cn/v1",
        api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
    )

    # question = input("输入你的问题：")
    # question = 'How does current academic research combine the Retrieval Augmented Generation(RAG) framework and large language models(LLMs) to improve the answer effect in downstream fields?'

    question = 'What drugs should be used to treat COVID-19? Will there be any adverse sequelae?'

    """

    keyword_prompt = f'Please extract 3 most relevant keywords based on the following question,' \
             f'You must respond with a list of strings in the following format: ["keyword 1", "keyword 2", "keyword 3"].' \
             f'Requirement: 1) The keywords should be used to search for relevant academic papers. ' \
             f'2) The keywords should summarize the main idea of the question. ' \
             f'3) The keywords can be supplements or extensions to the original question, not limited to the original question.' \
             f'Question below:{question}'

    keywords_content = get_keywords(keyword_prompt)
    print("key_words:", keywords_content)






    #arxiv用例
    #arxiv_url = new_get_url(keywords_content)
    # 保存论文爬虫结果至csv文件中
    start_page = 0
    end_page = 4
    scrape_arxiv(start_page, end_page, url, ori_file_path)
    df = pd.read_csv(ori_file_path)
    # 删除重复行，保留第一次出现的行
    df_unique = df.drop_duplicates(subset='ArXiv Number', keep='first')
    # 将清理后的数据保存到新的 CSV 文件
    df_unique.to_csv(new_file_path, index=False)

    # 进行初筛 rerank
    #rerank_path="./output/csv_download/arxivpa_Rerank_result.csv"
    # top_docs = self.rerank_batch(df_unique)
    # top_docs_df = pd.DataFrame(top_docs)
    # top_docs_df.to_csv(rerank_path, index=False)
    top_docs_df = df_unique




    for index, row in top_docs_df.iterrows():
        citation_number = get_pubmed_citation_Number(row['Arxiv Number'])
        top_docs_df.at[index, 'Citation Number'] = citation_number
        top_docs_df.to_csv(citation_number_path, index=False)

    # 根据被引数量进行排名
    max_citation_row = top_docs_df.loc[top_docs_df['Citation Number'].idxmax()]
    max_arxiv_number = max_citation_row['Arxiv Number']



    #max_arxiv_number=2312.10997
    max_citation_number = max_citation_row['Citation Number']
    max_abstract = max_citation_row['Abstract']
    max_url = f"https://arxiv.org/pdf/{max_arxiv_number}.pdf"
    # 保存被引量最多的文章 的被引和引用文献数据
    references_filename = reference_path + f"{max_arxiv_number}_references.csv"
    citations_filename = citation_path + f"{max_arxiv_number}_citations.csv"
    print("被引量最多的文章arxiv number:", max_arxiv_number)
    print("被引量为：", max_citation_number)
    get_references_citations(max_arxiv_number, citations_filename, references_filename)

    # 从被引文献中搜索和核心论文最相关的文章
    df_origi = pd.read_csv(references_filename)
    origi_references_list = df_origi['Semantic Scholar ID'].tolist()

    df_citations = pd.read_csv(citations_filename)

    similarity_output_path = similarity_output + f"{max_arxiv_number}_similarity_output.csv"

    with open(similarity_output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(
            ["Title", "Abstract", "ArXiv ID", "URL", "similarity"])

    for paperid in df_citations['Semantic Scholar ID']:
        get_similarity(paperid, origi_references_list, similarity_output_path)

    # 得到最后相关的url

    pdf_urls, abstract_list = extract_pdf_urls(similarity_output_path)
    #将核心论文加入pdf_urls
    pdf_urls.append(max_url)

    print("The last PDF URLs found:")
    print(pdf_urls)


    #根据最相关的url，调用chatpdf进行总结
    introduction = ""
    literature = ""
    discussion = ""
    conclusion = ""
    prompts_type = ["Introduction","Literature","Disscussion"]
    for pdf_url in pdf_urls:
        for prompt_type in prompts_type:
            start = pdf_url.rfind('/') + 1
            end = pdf_url.rfind('.pdf')
            arxiv_number = pdf_url[start:end]
            text=get_summarize(pdf_url,prompt_type=prompt_type)

            if prompt_type == "Introduction":
                introduction += text + "\n"
            elif prompt_type == "Literature":
                literature += text + "\n"
            elif prompt_type == "Disscussion":
                discussion += text + "\n"

    conclusion = introduction + literature + discussion



    types = ["Introduction","Literature","Disscussion","Conclusion"]
    context = {"Introduction":introduction,
               "Literature":literature,
               "Disscussion":discussion,
               "Conclusion":conclusion}

    reports = {}
    for type in types:
        text = context[type]
        reports[type] = generate_report(text, question, prompt_type=type)


    print("reports字典内容如下所示：")
    print(reports)

    raw_text = dict_to_text(reports)
    content = preprocess_text(raw_text)
    create_pdf(content, pdf_file_name)
    print(f"文件已保存为：{pdf_file_name}")
    """

    """
    #pubmed用例
    pubmed_url = create_pubmed_url(keywords_content)
    # 保存论文爬虫结果至csv文件中
    start_page = 1
    end_page = 5
    ori_file_path="./output/csv_download/arxivpa_original_result.csv"
    scrape_pubmed(start_page,end_page,pubmed_url,ori_file_path)
    df = pd.read_csv(ori_file_path)
    # 删除重复行，保留第一次出现的行
    df_unique = df.drop_duplicates(subset='DOI', keep='first')
    # 将清理后的数据保存到新的 CSV 文件
    new_file_path="./output/csv_download/arxivpa_afterFusion_result.csv"
    df_unique.to_csv(new_file_path, index=False)
    top_docs_df = df_unique


    # 得到被引数量
    citation_number_path="./output/csv_download/citation_ed_number.csv"
    for index, row in top_docs_df.iterrows():
        citation_number = get_pubmed_citation_Number(row['DOI'])
        top_docs_df.at[index, 'Citation Number'] = citation_number
        top_docs_df.to_csv(citation_number_path, index=False)

    # 根据被引数量进行排名
    max_citation_row = top_docs_df.loc[top_docs_df['Citation Number'].idxmax()]
    max_pubmed_DOI = max_citation_row['DOI']
    max_pmcid = get_pmcid_from_doi(max_pubmed_DOI)

    """
    max_pmcid = pmcid
    max_pubmed_DOI = doi

    with contextlib.redirect_stdout(io.StringIO()) as f:
        print("被引量最多的文章pmcid:", pmcid)
        print("被引量最多的文章doi:", doi)
        # 每次打印后立即发送输出内容到前端
        output = f.getvalue()
        emit('update_output', {'data': output})
        f.truncate(0)
        f.seek(0)

    # max_citation_number = max_citation_row['Citation Number']
    max_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{max_pmcid}/pdf/"
    # 保存被引量最多的文章 的被引和引用文献数据
    references_filename = reference_path + f"{max_pmcid}_references.csv"
    citations_filename = citation_path + f"{max_pmcid}_citations.csv"
    # print("被引量最多的文章pmcid:", max_pmcid)
    # print("被引量为：", max_citation_number)

    get_references_citations(citations_filename, references_filename,doi=max_pubmed_DOI)

    # 从被引文献中搜索和核心论文最相关的文章
    df_origi = pd.read_csv(references_filename)
    origi_references_list = df_origi['Semantic Scholar ID'].tolist()

    df_citations = pd.read_csv(citations_filename)

    similarity_output_path = similarity_output + f"{max_pmcid}_similarity_output.csv"

    with open(similarity_output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(["Title", "Abstract", "arxivID", "URL", "DOI", "similarity"])

    session = setup_session()
    for paperid in df_citations['Semantic Scholar ID']:
        get_similarity(session, paperid, origi_references_list, similarity_output_path)

    # 得到最后相关的url

    pdf_urls, abstract_list = extract_pdf_urls(similarity_output_path, type="pubmed")
    # 将核心论文加入pdf_urls
    pdf_urls.append(max_url)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        print("The last PDF URLs found:")
        print(pdf_urls)
        # 每次打印后立即发送输出内容到前端
        output = f.getvalue()
        emit('update_output', {'data': output})
        f.truncate(0)
        f.seek(0)


    # 根据最相关的url，调用chatpdf进行总结
    introduction = ""
    literature = ""
    discussion = ""
    conclusion = ""
    prompts_type = ["Introduction", "Literature", "Disscussion"]
    for pdf_url in pdf_urls:

        # 设置保存pdf的路径
        parts = pdf_url.split('/')
        pmc_id = parts[5]
        pdf_save_path = f"./output/pdf_download/top_pdf/{pmc_id}.pdf"
        download_pdf(pdf_url, pdf_save_path)  # 下载pdf文件

        for prompt_type in prompts_type:
            start = pdf_url.rfind('/') + 1
            end = pdf_url.rfind('.pdf')
            arxiv_number = pdf_url[start:end]
            text = get_summarize(pdf_url, prompt_type=prompt_type, filepath=pdf_save_path)

            if prompt_type == "Introduction":
                introduction += text + "\n"
            elif prompt_type == "Literature":
                literature += text + "\n"
            elif prompt_type == "Disscussion":
                discussion += text + "\n"

    conclusion = introduction + literature + discussion

    types = ["Introduction", "Literature", "Disscussion", "Conclusion"]
    context = {"Introduction": introduction,
               "Literature": literature,
               "Disscussion": discussion,
               "Conclusion": conclusion}

    reports = {}
    for type in types:
        text = context[type]
        reports[type] = generate_report(text, question, prompt_type=type)


    with contextlib.redirect_stdout(io.StringIO()) as f:
        print("reports字典内容如下所示：")
        print(reports)
        # 每次打印后立即发送输出内容到前端
        output = f.getvalue()
        emit('update_output', {'data': output})
        f.truncate(0)
        f.seek(0)


    raw_text = dict_to_text(reports)
    content = preprocess_text(raw_text)
    create_pdf(content, pdf_file_name)


    with contextlib.redirect_stdout(io.StringIO()) as f:
        print(f"文件已保存为：{pdf_file_name}")
        # 每次打印后立即发送输出内容到前端
        output = f.getvalue()
        emit('update_output', {'data': output})
        f.truncate(0)
        f.seek(0)

