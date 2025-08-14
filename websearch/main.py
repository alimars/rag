from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import PlaywrightURLLoader
from langdetect import detect
from bs4 import BeautifulSoup
from fastapi.openapi.utils import get_openapi
import requests ,time,random, os

app = FastAPI(title="DDG + LangChain Summarizer", version="1.1.0")
Summarizer_HOST = os.environ.get('Summarizer_HOST', 'localhost:11434')
class ToolInput(BaseModel):
    query: str
    num_results: int = 3

class ToolOutput(BaseModel):
    results: List[dict]

def ddg_search(query, num_results=3):
    print(f"[SEARCH] DuckDuckGo: {query}")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query}, headers=headers, timeout=10)

    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.select("a.result-link")
    urls = [link.get("href") for link in links if link.get("href") and link.get("href").startswith("http")]
    print(f"[SEARCH] Found {len(urls)} URLs, using top {num_results}")
    return urls[:num_results]

def summarize_with_ollama(text, model="llama3.2", max_tokens=400):
    try:
        print("[LLM] Summarizing content with Ollama...")
        resp = requests.post(f"http://{Summarizer_HOST}/api/generate", json={
            "model": model,
            "prompt": f"Summarize the following article:\n\n{text[:3000]}\n\nBe concise and clear.",
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }, timeout=20)

        summary = resp.json().get("response", "").strip()
        print(f"[LLM] Summary length: {len(summary)} characters")
        return summary

    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        return "[Summary failed]"

@app.post("/invoke", response_model=ToolOutput)
def invoke(input: ToolInput):
    urls = ddg_search(input.query, input.num_results)
    results = []

    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["nav", "footer", "header", "script", "style"]
    )

    try:
        print("[LOAD] Loading pages with Playwright...")
        documents = loader.load()
    except Exception as e:
        print(f"[ERROR] Page load failed: {e}")
        return {"results": [{"url": "", "content": "[Error loading pages]"}]}

    source_urls = []
    for doc in documents:
        url = doc.metadata.get("source", "")
        text = doc.page_content.strip()

        try:
            lang = detect(text[:500])
            if lang != "en":
                print(f"[SKIP] Non-English content from {url}")
                continue
        except:
            print(f"[WARN] Could not detect language, skipping {url}")
            continue

        print(f"[PROCESS] Summarizing {url}")
        summary = summarize_with_ollama(text)
        if summary == "[Summary failed]" or not summary.strip():
            print("[FALLBACK] Using first paragraph as fallback summary")
            summary = text.strip().split("\n")[0][:300] + "..."
            
        results.append({
            "url": url,
            "summary": summary,
            "content": text[:1500]
        })
        source_urls.append(url)
        time.sleep(random.uniform(1.5, 2.5))

    if not results:
        print("[RESULT] No usable English content found.")
        results.append({"url": "", "summary": "[No valid results]", "content": ""})

    return {"results": results, "source_urls": source_urls}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="DuckDuckGo + LangChain + Ollama summarizer",
        routes=app.routes
    )

    openapi_schema["x-openwebui"] = {
        "tool_type": "web-search",
        "enabled_by_default": True,
        "icon": "🧠"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
