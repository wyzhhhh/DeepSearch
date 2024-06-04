import requests

x_api_key = 'sec_wIIMRcjND18eu3DKWjYg3nngYaz9hta3'
from openai import OpenAI
import pandas as pd
import voyageai
import os
import openai
import uuid
from save_pdf import *

client = OpenAI(
    base_url="https://kapkey.chatgptapi.org.cn/v1",
    api_key="sk-46sQ2NQu5oOtoNc8416dB643BdA84151A204F44b3313Dd8d"
)

os.environ["VOYAGE_API_KEY"] = "pa-5C2ZvLnYfOYMIxzA8JD4k5jNn8gIIQIUxCOs5PpPJaE"
voyage_api_key = os.getenv('VOYAGE_API_KEY')
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch  # 导入inch

def rerank_batch(query, df):
    vo = voyageai.Client()
    batch_size = 500
    num_batches = (len(df) + batch_size - 1) // batch_size  # Calculate the number of batches
    top_documents = []

    for batch in range(num_batches):
        start_index = batch * batch_size
        end_index = min((batch + 1) * batch_size, len(df))
        batch_df = df.iloc[start_index:end_index]
        documents = batch_df['Abstract'].tolist()
        ids_titles = [(row['ArXiv Number'], row['Title']) for index, row in batch_df.iterrows()]

        reranking = vo.rerank(query, documents, model="rerank-lite-1", top_k=50)
        for r in reranking.results:
            print(f"Document: {r.document}")
            print(f"Relevance Score: {r.relevance_score}")
            index = documents.index(r.document)
            top_documents.append({
                "ArXiv Number": ids_titles[index][0],
                "Title": ids_titles[index][1],
                "Abstract": r.document
            })

    return top_documents




def generate_report(context,question,prompt_type,paper,relatedPaper,researchProblem="",researchProblemRationale="",scientificMethod="",scientificMethodRationale="",report_format="apa"):





    introduction_prompt = "You are tasked with generating a comprehensive INTRODUCTION of a report based on the information provided. This INTRODUCTION should be thorough, well-structured, informative, and provide in-depth analysis relevant to the query at hand. Utilize all the information provided to craft an INTRODUCTION section of the report that not only answers the query but also provides an in-depth look at the importance and contextual significance of the topic. The aim is to produce an introduction to the report that highlights the background significance and information context of the research topic. " \
                "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                "- The introduction is your first section. Start with explaining the background, importance, purpose, and scope of the research topic." \
                "- Clearly define the scope of the research and elaborate on why the topic merits attention." \
                "- Develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                "Your INTRODUCTION should adhere strictly to the specified format. Additionally, you are expected to:" \
                "- Include all used source URLs at the end of the report as references. Ensure no duplication of sources; only one reference for each source is needed." \
                "- Cite search results within the text using inline citations. Select only the most relevant results that directly contribute to answering the query." \
                "Upon completion, ensure your opinions are well-founded and clearly articulated based on the information given. Avoid general or inconclusive statements, and make certain that your citations are accurately placed at the end of sentences or paragraphs where they are referenced." \
                "I am going to provide the related papers context ,question as follows:" \




    discussion_prompt = "You are tasked with generating a comprehensive DISCUSSION of areport based on the information provided. This DISCUSSION should be thorough, well-structured, informative, and provide in-depth analysis relevant to the query at hand. Utilize all the information provided to create a DISCUSSION of the report that not only answers the query but also provides an in-depth look at the pros and cons of approaches to the topic. The aim is to produce a discussion section of the report that highlights a critical analysis of the results of the research topic while establishing a clear and valid personal perspective supported by the data. " \
                    "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                    "- The discussion is your section. Start with synthesizing the information in the literature review and pointing out the interrelationships, trends, and potential contradictions between studies." \
                    "- Discuss unresolved problems, research gaps, and future research directions in the research field." \
                    "- Synthesize the information in the literature review and propose a scientific method to solve the research problem. Your method should be clear, innovative, rigorous, effective, and generalizable. This will be based on a deep understanding of the research problem, its principles, existing research." \
                    "You must develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                    "Your report should adhere strictly to the specified format. Additionally, you are expected to:" \
                    "- Include all used source URLs at the end of the report as references. Ensure no duplication of sources; only one reference for each source is needed." \
                    "- Cite search results within the text using inline citations. Select only the most relevant results that directly contribute to answering the query." \
                    "Upon completion, ensure your opinions are well-founded and clearly articulated based on the information given. Avoid general or inconclusive statements, and make certain that your citations are accurately placed at the end of sentences or paragraphs where they are referenced." \
                    "You are required to write the report in the following format, ensuring all sections are comprehensive and detailed." \
                    "I am going to provide the related papers context, question as follows:" \





    literature_prompt = "You are tasked with generating a comprehensive LITERATURE REVIEW section of report based on the information provided. This LITERATURE REVIEW should be thorough, well-structured, informative, and provide in-depth analysis relevant to the query at hand. Use all the information provided to produce a LITERATURE REVIEW that not only answers the query but also provides an in-depth look at the literature on the topic. The aim is to produce a literature review that highlights a summary of existing research relevant to the topic, while establishing a clear and valid personal perspective supported by data. " \
                    "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                    "- The literature review is your section. Begin by providing a detailed review and discussion of the literature mentioned in the above information, including experimental methods, experimental settings, experimental results, etc." \
                    "- Analyze the trends, successes, and shortcomings of existing research, which may include comparisons of theories, methodologies, or findings." \
                    "You must develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                    "Your report should adhere strictly to the specified format. Additionally, you are expected to:" \
                    "- Include all used source URLs at the end of the report as references. Ensure no duplication of sources; only one reference for each source is needed." \
                    "- Cite search results within the text using inline citations. Select only the most relevant results that directly contribute to answering the query." \
                    "Upon completion, ensure your opinions are well-founded and clearly articulated based on the information given. Avoid general or inconclusive statements, and make certain that your citations are accurately placed at the end of sentences or paragraphs where they are referenced." \
                    "You are required to write the report in the following format, ensuring all sections are comprehensive and detailed." \
                    "I am going to provide the related papers context, question as follows:" \






    for i, ctx in enumerate(context):
        introduction_prompt += f"The INTRODUCTION section is below: {ctx['Introduction']}\n The source url is :{ctx['url']}  "
        discussion_prompt += f"The DISCUSSION section is below:\n {ctx['Discussion']}\n The source url is :{ctx['url']}   "
        literature_prompt += f"The LITERATURE section is below:\n {ctx['Literature']}\n The source url is :{ctx['url']}   "




    introduction_prompt += f"Related papers abstracts: {relatedPaper['abstracts']} " \
                        f"The research question is: {question} " \
                        "Then, following your review of the above content, please proceed to generate an Introduction section of a paper, in the format of" \
                        "Introduction: "

    discussion_prompt +=  f"Related papers abstracts:{relatedPaper['abstracts']}" \
                        f"The research question is :{question}" \
                        "Then, following your review of the above content, please proceed to generate a Discussion section of a paper, in the format of" \
                        "Discussion:"

    literature_prompt += f"Related papers abstracts: {relatedPaper['abstracts']}" \
                        f"The research question is: {question}" \
                        "Then, following your review of the above content, please proceed to generate a Literature Review section of a paper, in the format of" \
                        "Literature Review:"





    conclusion_prompt = "You are tasked with generating a comprehensive conclusion of a report based on the information provided. This section should be thorough, well-structured, informative, and provide in-depth analysis relevant to the query at hand. Use all the information provided to craft a CONCLUSION of a report that not only answers the query but also provides insight into the significance of the results on the topic. The purpose is to produce a CONCLUSION that highlights final reflections on the entire study while establishing a clear and valid personal perspective supported by the data." \
                    "Preparation for the report involves understanding the provided information, which serves as the basis for addressing the specific query or task. Here’s how to proceed:" \
                    "- The conclusion is your final section. Begin by summarizing the key points based on all of the above information." \
                    "You must develop and articulate a concrete, reasoned opinion based on the provided information, steering clear of vague or unsubstantiated conclusions." \
                    "Your report should adhere strictly to the specified format. Additionally, you are expected to:" \
                    "- Include all used source URLs at the end of the report as references. Ensure no duplication of sources; only one reference for each source is needed." \
                    "- Cite search results within the text using inline citations. Select only the most relevant results that directly contribute to answering the query." \
                    "Upon completion, ensure your opinions are well-founded and clearly articulated based on the information given. Avoid general or inconclusive statements, and make certain that your citations are accurately placed at the end of sentences or paragraphs where they are referenced." \
                    "You are required to write the report in the following format, ensuring all sections are comprehensive and detailed." \
                    "I am going to provide the related papers context, question as follows:" \
                    f"Target paper title: {paper['title']}" \
                    f"Target paper abstract: {paper['abstract']}" \
                    f"Related paper titles: {relatedPaper['titles']}" \
                    f"Related paper abstracts: {relatedPaper['abstracts']}" \
                    f"The research question is: {question}" \
                    "Then, following your review of the above content, please proceed to generate the conclusion of a paper, in the format of" \
                    "Conclusion:"





    idea_prompt="You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the target paper, those of" \
                "related papers in the existing literature potentially connected to the research area." \
                "Understanding of the target paper, related papers is essential:" \
                "- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem." \
                "- The related papers are studies that have cited the target paper, indicating their direct relevance and connection to the primary research topic you are focusing on, and providing additional context and insights that are essential for understanding and expanding upon the target paper." \
                "Your approach should be systematic:" \
                "- Start by thoroughly reading the title and abstract of the target paper to understand its core focus." \
                "- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective and insights relevant to the primary research topic." \
                "- Finally, explore to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant." \
                "I am going to provide the target paper, related papers as follows:" \
                f"Target paper title: {paper['title']}" \
                f"Target paper abstract: {paper['abstract']}" \
                f"Related paper titles: {relatedPaper['titles']}" \
                f"Related paper abstracts: {relatedPaper['abstracts']}" \
                "With the provided target paper, related papers, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process." \
                "Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of" \
                "Problem:" \
                "Rationale: "


    #researchProblem="Exploring the Role of Asymptomatic Carriers in SARS-CoV-2 Transmission Dynamics"
    #researchProblemRationale="The investigation of the first known person-to-person transmission of SARS-CoV-2 in the USA highlighted the significance of asymptomatic carriers in the spread of COVID-19. Despite active symptom monitoring and testing, no further transmission occurred from the identified asymptomatic contacts. This raises the question of the role of asymptomatic carriers in SARS-CoV-2 transmission dynamics. Understanding the infectivity and transmission patterns of asymptomatic carriers is crucial for devising effective public health strategies, including targeted testing, contact tracing, and containment measures. This research problem aims to elucidate the contribution of asymptomatic carriers to the transmission of SARS-CoV-2 and assess the effectiveness of current surveillance and control measures in mitigating the spread of COVID-19."

    method_prompt = "You are going to propose a scientific method to address a specific research problem. Your method should be clear, innovative, rigorous, valid, and generalizable. " \
                    "This will be based on a deep understanding of the research problem, its rationale, existing studies." \
                    "Understanding of the research problem, existing studies is essential:" \
                    "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities, which should be the cornerstone of your method development." \
                    "- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the problem discovery phase, all serving as foundational material for developing the method." \
                    "Your approach should be systematic:" \
                    "- Start by thoroughly reading the research problem and its rationale, to understand your primary focus." \
                    "- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic." \
                    "- Finally, explore to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant." \
                    "I am going to provide the research problem, existing studies (target paper & related papers), as follows:" \
                    f"Research problem: {researchProblem}" \
                    f"Rationale: {researchProblemRationale}" \
                    f"Target paper title: {paper['title']}" \
                    f"Target paper abstract: {paper['abstract']}" \
                    f"Related paper titles: {relatedPaper['titles']}" \
                    f"Related paper abstracts: {relatedPaper['abstracts']}" \
                    "With the provided research problem, existing studies, your objective now is to formulate a method that not only leverages these resources but also strives to be clear, innovative,rigorous, valid, and generalizable. Before crafting the method, revisit the research problem, to ensure it remains the focal point of your method development process." \
                    f"Research problem: {researchProblem}" \
                    f"Rationale: {researchProblemRationale}" \
                    "Then, following your review of the above content, please proceed to propose your method with its rationale, in the format of" \
                    "Method:" \
                    "Rationale:"


    experiment_prompt = "You are going to design an experiment, aimed at validating a proposed method to address a specific research problem. " \
                        "Your experiment design should be clear, robust, reproducible, valid, and feasible. This will be based on a deep understanding of the research problem, scientific method, existing studies." \
                        "Understanding of the research problem, scientific method, existing studies is essential:" \
                        "- The research problem has been formulated based on an in-depth review of existing studies and a potential exploration of relevant entities." \
                        "- The scientific method has been proposed to tackle the research problem, which has been informed by insights gained from existing studies and relevant entities." \
                        "- The existing studies refer to the target paper that has been pivotal in identifying the problem and method, as well as the related papers that have been additionally referenced in the discovery phase of the problem and method, all serving as foundational material for designing the experiment." \
                        "Your approach should be systematic:" \
                        "- Start by thoroughly reading the research problem and its rationale followed by the proposed method and its rationale, to pinpoint your primary focus." \
                        "- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective and insights relevant to the primary research topic." \
                        "- Finally, explore to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant." \
                        "I am going to provide the research problem, scientific method, existing studies (target paper & related papers), as follows:" \
                        f"Research problem: {researchProblem}" \
                        f"Rationale: {researchProblemRationale}" \
                        f"Scientific method: {scientificMethod}" \
                        f"Rationale: {scientificMethodRationale}" \
                        f"Target paper title: {paper['title']}" \
                        f"Target paper abstract: {paper['abstract']}" \
                        f"Related paper titles: {relatedPaper['titles']}" \
                        f"Related paper abstracts: {relatedPaper['abstracts']}" \
                        "With the provided research problem, scientific method, existing studies, your objective now is to design an experiment that not only leverages these resources but also strives to be clear, robust, reproducible, valid, and feasible. Before crafting the experiment design, revisit the research problem and proposed method, to ensure they remain at the center of your experiment design process." \
                        f"Research problem: {researchProblem}" \
                        f"Rationale: {researchProblemRationale}" \
                        f"Scientific method: {scientificMethod}" \
                        f"Rationale: {scientificMethodRationale}" \
                        "Then, following your review of the above content, please proceed to outline your experiment with its rationale, in the format of" \
                        "Experiment:" \
                        "Rationale:"

    system_content="You are a helpful assistant."
    if prompt_type == "Introduction":
        system_content = "You are an AI assistant whose main goal is to summarize the research background and significance of the topic based on the given context, so as to help researchers write the Introduction section of the paper."
        prompt = introduction_prompt
    elif prompt_type == "Literature":
        system_content = "You are an AI assistant whose main goal is to summarize the research status and progress of the topic based on the given context to help researchers write the Literature Review section of the paper."
        prompt = literature_prompt
    elif prompt_type == "Discussion":
        system_content = "You are an AI assistant whose main goal is to summarize the advantages and disadvantages of current research based on the given context to help researchers write the Literature Review section of their paper."
        prompt = discussion_prompt
    elif prompt_type == "Conclusion":
        system_content = "You are an AI assistant whose main goal is to summarize the given context in a way that researchers can use to write the Literature Review section of their paper."
        prompt = conclusion_prompt
    elif prompt_type == "idea":
        system_content = "You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on existing scientific literature, in order to aid researchers in discovering novel and significant research opportunities that can advance the field."
        prompt = idea_prompt
    elif prompt_type == "method":
        system_content = "You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid methodologies to solve newly identified scientific problems derived from existing scientific literature, in order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in their fields."
        prompt = method_prompt
    elif prompt_type == "experiment":
        system_content = "You are an AI assistant whose primary goal is to design robust, feasible, and impactful experiments based on identified scientific problems and proposed methodologies from existing scientific literature, in order to enable researchers to systematically test hypotheses and validate groundbreaking discoveries that can transform their respective fields."
        prompt = experiment_prompt
    print(f"-----------------------------{prompt_type}------------------------------")
    print(prompt)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": prompt}],
        temperature=0,
    )

    if completion.choices:
        # 提取第一个选项
        first_choice = completion.choices[0]
        # 提取该选项的消息内容
        report = first_choice.message.content
        print("----------------------------------------report----------------------------------------")
        print(report)
        return report

        # 将生成的文本保存为PDF文件
        # 设置PDF文件路径和名称

    else:
        print("No choices found.")








"""

file_path = "./new_arxiv.csv"  # Make sure to set your actual file path
df = pd.read_csv(file_path)
query = "Please give specific academic research reports on the intelligent answer systems in various vertical fields that Retrieval-Augment-Generate(RAG) combined with LLM and knowledge graph construction."
top_docs = rerank_batch(query, df)

# Convert the list of dictionaries to a DataFrame and save it to a CSV file
top_docs_df = pd.DataFrame(top_docs)
top_docs_df.to_csv("reranked_documents.csv", index=False)


import arxiv

search = arxiv.Search(
    query = "reranker",
    max_results = 2,
    sort_by = arxiv.SortCriterion.Relevance
)
#search_response = []
for result in search.results():
    print(result.entry_id, '->', result.title)
    print(result.links)

def get_references(doi):
    url = f"http://api.crossref.org/works/{doi}"
    response = requests.get(url)
    data = response.json()
    references = data.get('message', {}).get('reference', [])
    for index, ref in enumerate(references, start=1):
        print(f"{index}: {ref.get('unstructured', 'No reference found')}")


# 示例DOI
doi = "10.1038/nature12373"
get_references(doi)
"""
