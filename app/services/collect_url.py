import urllib.request
import datetime
import json
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import os

# Load API credentials from environment variables or config file
CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "ZdXlVrCKm3aVo2CK8HzC")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "fWGgWzZPhJ")
# MODEL_PATH = 

llm = LlamaCpp(
    model_path="app/Magistral-Small-2506-UD-IQ2_XXS.gguf", 
    verbose=True,
    n_ctx=2048,
    n_batch=16,
)

# Define the prompt template and model for generating search terms
# template = """Do not answer the following questions directly. Instead, list at least 3 and up to 10 Korean keywords related to the questions and covering stock-related topics, separated by commas. Do not write any other sentences; just list the keywords.

# # question: {question}
# # """

# prompt = PromptTemplate.from_template(template)

# chain = prompt | llm

# Function to request data from Naver API
def get_request_url(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", CLIENT_ID)
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"[{datetime.datetime.now()}] HTTP Error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        print(f"[{datetime.datetime.now()}] URL Error: {e.reason}")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Unexpected Error: {e}")

    return None


# Function to fetch search results from Naver API
def get_naver_search_result(sNode, search_text, page_start, display):
    base_url = "https://openapi.naver.com/v1/search"
    node = f"/{sNode}.json"
    parameters = f"?query={urllib.parse.quote(search_text)}&start={page_start}&display={display}&sort=date"
    url = base_url + node + parameters

    response_data = get_request_url(url)
    return json.loads(response_data) if response_data else None


# Function to extract data from a post and add it to the results list
def get_post_data(post, json_result):
    title = post["title"]
    description = post["description"]
    org_link = post["originallink"]
    link = post["link"]

    # Parse publication date
    pDate = datetime.datetime.strptime(post["pubDate"], "%a, %d %b %Y %H:%M:%S +0900")

    # Filter posts published within the last 6 months
    six_months_ago = datetime.datetime.now() - datetime.timedelta(days=6 * 30)

    if pDate >= six_months_ago:
        # Exclude articles mentioning outdated years
        if "2022" in title or "2022" in description:
            print(f"Excluded outdated article: {title}")
        else:
            json_result.append(
                {
                    "title": title,
                    "description": description,
                    "org_link": org_link,
                    "link": link,
                    "pDate": pDate.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )


# Function to collect links from the JSON result
def collect_links(json_result):
    link_data = []
    for item in json_result:
        link_data.append(item["link"])
        link_data.append(item["org_link"])
    return link_data


import datetime


def get_urls_from_question(question):
    global url_list
    sNode = "news"
    display_count = 100
    url_list = []  # Initialize url_list for each new question
    
    

    # response = chain.invoke({"question": question})
    # print(response)  # 함수 안에서 찍음 — OK

    # llama_words = [item.strip() for item in response.split(",")]
    # filtered_keywords = [kw for kw in llama_words if "2022" not in kw]
    
    splt_word = question.split()

    for search_text in splt_word:
        print(f"Processing search for: {search_text}")
        json_result = []
        json_search = get_naver_search_result(sNode, search_text, 1, display_count)
        while json_search and json_search.get("display", 0) > 0:
            for post in json_search["items"]:
                get_post_data(post, json_result)
            next_start = json_search.get("start", 1) + json_search.get("display", 0)
            json_search = get_naver_search_result(sNode, search_text, next_start, display_count)

        url_list.extend(collect_links(json_result))

    return url_list