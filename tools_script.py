'''
Script to define various tools for a language model agent.
'''

## --- Import necessary libraries ---
import requests, tempfile, base64, json, os, fitz, re, pandas as pd, datetime
from langchain.tools import tool
from bs4 import BeautifulSoup
# import yt_dlp
from langchain_community.utilities import ArxivAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv


# Load the environment variables
load_dotenv()
HF_ACCESS_KEY = os.getenv('HF_ACCESS_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')


## --- TOOL 1: Wikipedia Summary Tool ---
@tool
def wikipedia_and_generalknowledge_search(query: str) -> str:
    """
    Answer questions related to general knowledge, world information, facts, sports, olympics, history, etc. from Wikipedia by scraping the text and returns text as context for LLM to use.
    Input: question as string
    Output: extracted text
    """
    # Step 1: Search Wikipedia for the most relevant page
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    try:
        resp = requests.get(search_url, params=params, timeout=150)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return "no_answer"
        page_title = results[0]["title"]
        page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    except Exception:
        return "error: Could not search Wikipedia"

    # Step 2: Fetch the Wikipedia page and extract main text
    try:
        page_resp = requests.get(page_url, timeout=120)
        page_resp.raise_for_status()
        soup = BeautifulSoup(page_resp.text, "html.parser")
        output = f"Source: {page_url}\n"

        # Extract main text from all paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        # Limit to first 3000 characters for brevity
        output += text[:3000] if text else "No textual content found."
        return output
    except Exception as e:
        return f"error: {e}"

# when you use the @tool decorator from langchain.tools, the tool.name and tool.description are automatically extracted from your function
# tool.name is set to the function name (e.g., `search_tool`), and 
# tool.description is set to the docstring of the function  (the triple-quoted string right under def ...) (e.g., "Answer general knowledge or current events queries using DuckDuckGo.").


# --- TOOL 2: Calculator Tool ---
@tool
def math_calculator(expression: str) -> str:
    """Evaluate math expressions."""
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "error"
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception:
        return "error"


# --- TOOL 3: Unit Conversion Tool ---
@tool
def convert_units(query: str) -> str:
    """
    Convert between metric and imperial units (length, mass, temperature).
    Input format: '<value> <from_unit> to <to_unit>', e.g. '10 meters to feet'
    """
    try:
        parts = query.lower().split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3]
        conversions = {
            ("meters", "feet"): lambda v: v * 3.28084,
            ("feet", "meters"): lambda v: v / 3.28084,
            ("kg", "lb"): lambda v: v * 2.20462,
            ("lb", "kg"): lambda v: v / 2.20462,
            ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        }
        func = conversions.get((from_unit, to_unit))
        if func:
            return str(round(func(value), 2))
        return "error"
    except Exception:
        return "error"


# --- TOOL 4: Date & Time Tool ---
@tool
def get_time(input: str) -> str:
    """Get current UTC time as HH:MM."""
    return datetime.datetime.utcnow().strftime("%H:%M")

@tool
def get_date(input: str) -> str:
    """Get current date as YYYY-MM-DD."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


# --- TOOL 5: Dictionary Tool ---
@tool
def dictionary_lookup(word: str) -> str:
    """
    Look up the definition of an English word using DictionaryAPI.dev.
    Returns a short definition or an error message if not found.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and "meanings" in data[0]:
            meanings = data[0]["meanings"]
            if meanings and "definitions" in meanings[0]:
                defs = meanings[0]["definitions"]
                if defs and "definition" in defs[0]:
                    return defs[0]["definition"]

        return f"No definition found for '{word}'."

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return f"No entry found for '{word}'."
        return f"HTTP error: {e}"
    except Exception as e:
        return f"Lookup error: {e}"

    
# --- TOOL 6: Currency Conversion Tool ---
@tool
def currency_converter(query: str) -> str:
    """
    Convert currencies using live exchange rates.
    Accepts input like: '100 USD to EUR' or '250 INR to GBP'.
    Returns the converted amount using exchangerate.host.
    """
    try:
        parts = query.upper().split()
        amount = float(parts[0])
        from_currency = parts[1]
        to_currency = parts[3]
        url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return str(round(data["result"], 2))
    except Exception:
        return "error"



# --- TOOL 7: Web Search Tool (DuckDuckGo) ---
@tool
def current_events_news_search_tool(query: str) -> str:
    """
    General web search tool for current events, news, or trending topics not yet on Wikipedia.
    Input: Input question
    Output: relevant context and source URL if available.
    """
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Check main answer fields
        for key in ["AbstractText", "Answer", "Definition"]:
            if data.get(key):
                answer = data[key].strip()
                break
        else:
            answer = None

        # Try to extract more from RelatedTopics
        if not answer:
            related = data.get("RelatedTopics")
            if related and isinstance(related, list):
                for topic in related:
                    if isinstance(topic, dict) and topic.get("Text"):
                        answer = topic["Text"].strip()
                        # Optionally, add the URL
                        if topic.get("FirstURL"):
                            answer += f"\nSource: {topic['FirstURL']}"
                        break

        # Try to extract from Results
        if not answer:
            results = data.get("Results")
            if results and isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and result.get("Text"):
                        answer = result["Text"].strip()
                        if result.get("FirstURL"):
                            answer += f"\nSource: {result['FirstURL']}"
                        break

        # Fallback: return "no_answer"
        if answer:
            return answer
        return "no_answer"
    except Exception as e:
        return f"error: {e}"

# --- TOOL 8: Weather Tool (OpenWeatherMap) ---
@tool
def get_weather(city: str) -> str:
    """Get current temperature in Celsius for a city."""
    import os
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return str(round(data["main"]["temp"]))
    except Exception:
        return "error"
   
# --- TOOL 9: Optical Character Recognition (OCR) Tool ---
@tool
def ocr_image(image_url: str) -> str:
    """
    Extracts all readable text from an image using HuggingFace TrOCR (microsoft/trocr-base-stage1).
    Input: URL to an image (e.g., PNG or JPG).
    Output: Recognized text string.
    """
    api_url = "https://api-inference.huggingface.co/models/microsoft/trocr-base-stage1"
    headers = {
        "Authorization": f"Bearer {HF_ACCESS_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": image_url}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data[0]["generated_text"]
    except Exception as e:
        return f"OCR error: {e}"

    
# --- TOOL 10: Image Classification Tool ---
@tool
def clasify_describe_image(image_url: str) -> str:
    """
    Generates a caption describing the contents of an image using HuggingFace (ViT-GPT2). 
    Use this tool to identify the main subject of an image so that an LLM can use it to answer further.
    Input: image URL
    Output: caption like 'A golden retriever lying on a couch.'
    """
    api_url = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    headers = {"Authorization": f"Bearer {HF_ACCESS_KEY}"}

    try:
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
        image_bytes = img_resp.content

        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else "no_caption"
    except Exception as e:
        return f"caption error: {e}"


# --- TOOL 11: Web Scraping Tool ---
@tool
def URL_scrape_tool(url: str) -> str:
    """
    Scrape the main textual content from a given website URL and returns the text - to be used as context by model.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WebScrapeTool/1.0)"
        }
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Try to extract main content from common tags
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        # Limit to first 2000 characters for brevity
        return text[:4000] if text else "No textual content found."
    except Exception as e:
        return f"error: {e}"

# --- TOOL 12: Audio to Text Transcription Tool ---
@tool
def audio_url_to_text(audio_url: str) -> str:
    """
    Transcribe speech from an audio file URL to text using Hugging Face's Whisper model.
    Input: A direct link to an audio file (e.g., .mp3, .wav).
    Output: The transcribed text.
    """
    api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {HF_ACCESS_KEY}"}
    try:
        # Download the audio file
        audio_resp = requests.get(audio_url, timeout=120)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content
        # Encode audio as base64 for API
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "inputs": audio_b64,
            "parameters": {"return_timestamps": False}
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "no_answer")
    except Exception as e:
        return f"error: {e}"

# --- TOOL 13: Python Code Executor Tool ---
@tool
def python_executor(code: str) -> str:
    """
    Safely execute simple Python code and return the result if the code is in the question. If the question has .py file attached, use 'python_excel_audio_video_attached_file_tool' tool first.
    Only supports expressions and basic statements (no imports, file I/O, or system access).
    """
    try:
        # Restrict built-ins for safety
        allowed_builtins = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range}
        # Only allow expressions, not statements
        result = eval(code, {"__builtins__": allowed_builtins}, {})
        return str(result)
    except Exception as e:
        return f"error: {e}"


# --- TOOL 14: YouTube Transcript Tool ---
@tool
def youtube_transcript_tool(video_url: str) -> str:
    """
    Get transcript (if available) for a YouTube video without downloading audio.
    Works only if subtitles or auto-captions exist.
    """
    try:
        # Extract video ID
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url)
        if not match:
            return "Invalid YouTube URL."
        video_id = match.group(1)

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([chunk['text'] for chunk in transcript])
        return full_text[:5000]  # truncate to keep LLM input manageable
    except Exception as e:
        return f"Transcript error: {e}"

# --- Tool 15: Search ArXiv portal for research papers information ----
@tool
def research_paper_search(query: str) -> str:
    """
    Search arXiv for journals/research/technical papers matching a query. 
    Returns top results including title, authors, abstract, and PDF link.
    """
    wrapper = ArxivAPIWrapper(
        top_k_results=2,                  # how many papers to return
        doc_content_chars_max=2000        # max chars of abstract to show
    )
    
    results_text = wrapper.run(query)
    return results_text


##-- Tool Discovery ---
# Use @tool for each function.
# Use get_all_tools() to auto-discover all decorated tools.
# tools_list = get_all_tools()
TOOLS_LIST = [
    wikipedia_and_generalknowledge_search,
    research_paper_search,
    python_executor,
    get_weather,
    math_calculator,
    convert_units,
    get_time,
    get_date,
    dictionary_lookup,
    currency_converter,
    ocr_image,
    clasify_describe_image,
    current_events_news_search_tool,
    URL_scrape_tool,
    audio_url_to_text, 
    youtube_transcript_tool,
]

TOOLS_DESCRIPTIONS = "\n".join(f"- {tool.name}: {tool.description}" for tool in TOOLS_LIST)