import streamlit as st
from portia import Portia, Config, PlanRunState, Tool, ToolRunContext, MultipleChoiceClarification
import os
from dotenv import load_dotenv
import logging
import requests
import ollama

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Clarification Class ---
class ContentReviewClarification(MultipleChoiceClarification):
    def __init__(self, content: dict, reason: str):
        super().__init__(
            category="content_review",
            user_guidance=f"Review this content: {content.get('title', 'Untitled')} - Reason: {reason}",
            options=["approve", "reject", "edit"]
        )
        self.content = content
        self.reason = reason

# --- Tool Class ---
class NewsFetchTool(Tool):
    def __init__(self):
        super().__init__(
            id="news_fetch",
            name="news_fetch",
            description="Fetch news articles based on keywords using NewsAPI."
        )

    def run(self, context: ToolRunContext, keywords: str) -> list:
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            raise ValueError("NEWSAPI_KEY not set in .env")
        url = f"https://newsapi.org/v2/everything?q={keywords}&apiKey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'ok':
                raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return data.get('articles', [])
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ValueError("NewsAPI authentication failed: Check NEWSAPI_KEY in .env or get a valid key at https://newsapi.org")
            raise ValueError(f"NewsAPI request failed: {str(e)}")

# --- Agent Logic ---
def validate_api_key(key: str, url: str, name: str) -> bool:
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        logger.debug(f"{name} API key validated successfully: {response.status_code}")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"{name} API key validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"{name} API connection error: {str(e)}")
        return False

def validate_newsapi_key(key: str) -> bool:
    try:
        response = requests.get(f"https://newsapi.org/v2/everything?q=test&apiKey={key}")
        response.raise_for_status()
        logger.debug(f"NewsAPI key validated successfully: {response.status_code}")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"NewsAPI key validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"NewsAPI connection error: {str(e)}")
        return False

def validate_ollama() -> bool:
    try:
        response = ollama.list()
        if 'llama3' in [model['name'] for model in response.get('models', [])]:
            logger.debug("Ollama llama3 model validated successfully")
            return True
        logger.error("Ollama llama3 model not found")
        return False
    except Exception as e:
        logger.error(f"Ollama connection error: {str(e)}")
        return False

def run_curation(user_prefs: dict):
    try:
        # Validate API keys and Ollama
        portia_key = os.getenv('PORTIA_API_KEY')
        newsapi_key = os.getenv('NEWSAPI_KEY')
        if not portia_key:
            raise ValueError("PORTIA_API_KEY is missing or empty in .env. Get a valid key at https://app.portialabs.ai")
        if not newsapi_key:
            raise ValueError("NEWSAPI_KEY is missing or empty in .env. Get a valid key at https://newsapi.org")

        # Preflight checks
        logger.debug("Validating connections before running task")
        if not validate_api_key(portia_key, "https://api.portialabs.ai/health", "Portia"):
            raise ValueError("PORTIA_API_KEY is invalid. Test with: curl -H 'Authorization: Bearer $PORTIA_API_KEY' https://api.portialabs.ai/health")
        if not validate_newsapi_key(newsapi_key):
            raise ValueError("NEWSAPI_KEY is invalid. Test with: curl 'https://newsapi.org/v2/everything?q=test&apiKey=$NEWSAPI_KEY'")
        if not validate_ollama():
            raise ValueError("Ollama llama3 model not found. Install Ollama, pull llama3 with 'ollama pull llama3', and run 'ollama serve'.")

        logger.debug("Initializing Portia with Ollama configuration")
        config = Config.from_default(
            llm_provider="ollama",
            llm_model="llama3",
            api_key=portia_key
        )

        tools = [NewsFetchTool()]
        portia = Portia(config=config, tools=tools)

        logger.debug(f"Running task with keywords: {user_prefs['keywords']}, threshold: {user_prefs['threshold']}")
        task = f"Fetch news articles on '{user_prefs['keywords']}'. Analyze each for sentiment using Ollama llama3. Flag if compound sentiment < {user_prefs['threshold']} or potentially misleading/offensive (e.g., bias, controversy). If flagged, raise clarification for human review."

        plan_run = portia.run(task)

        curated = plan_run.outputs.final_output.get('articles', []) if plan_run.state != PlanRunState.NEED_CLARIFICATION else []
        logger.debug("Curation completed successfully")
        return plan_run, curated
    except Exception as e:
        logger.error(f"Agent error occurred: {str(e)}")
        if "401" in str(e) or "Unauthorized" in str(e):
            raise ValueError("API authentication failed: Check PORTIA_API_KEY (https://app.portialabs.ai) and NEWSAPI_KEY (https://newsapi.org). Test Portia: curl -H 'Authorization: Bearer $PORTIA_API_KEY' https://api.portialabs.ai/health")
        raise ValueError(f"Agent error: {str(e)}")

# --- Streamlit UI ---
load_dotenv()

st.title("Human-Verified Content Curation Agent 🤖✅")
st.markdown("<style>h1 {color: #1E90FF;}</style>", unsafe_allow_html=True)

keywords = st.text_input("Keywords (e.g., AI innovation)", value="AI")
threshold = st.slider("Sentiment Threshold", -1.0, 1.0, 0.1)

if "plan_run" not in st.session_state:
    st.session_state.plan_run = None
if "curated" not in st.session_state:
    st.session_state.curated = []
if "outstanding_clarifications" not in st.session_state:
    st.session_state.outstanding_clarifications = []

if st.button("Curate Content") and keywords:
    with st.spinner("Curating..."):
        try:
            st.session_state.plan_run, st.session_state.curated = run_curation({"keywords": keywords, "threshold": threshold})
            st.session_state.outstanding_clarifications = st.session_state.plan_run.get_outstanding_clarifications() if st.session_state.plan_run.state == PlanRunState.NEED_CLARIFICATION else []
        except ValueError as e:
            st.error(f"Error: {str(e)}. If a 401 Unauthorized error, verify PORTIA_API_KEY and NEWSAPI_KEY in .env.")

if st.session_state.outstanding_clarifications:
    st.header("Human Review Needed")
    for idx, clar in enumerate(st.session_state.outstanding_clarifications):
        st.write(clar.user_guidance)
        action = st.selectbox("Action", clar.options, key=f"action_{idx}")
        edited_desc = ""
        if action == "edit":
            edited_desc = st.text_input("Edited Description", key=f"edit_{idx}", value=clar.content.get('description', ''))
        if st.button("Submit Review", key=f"submit_{idx}"):
            updated_content = clar.content.copy()
            if action == "edit":
                updated_content['description'] = edited_desc
            st.session_state.plan_run = st.session_state.plan_run.resolve_clarification(clar, action)
            if action != "reject":
                st.session_state.curated.append(updated_content)
            st.session_state.plan_run = st.session_state.plan_run.resume()
            st.session_state.outstanding_clarifications = st.session_state.plan_run.get_outstanding_clarifications()
            st.rerun()

if st.session_state.curated:
    st.header("Curated Content")
    for item in st.session_state.curated:
        st.write(f"**{item.get('title', 'Untitled')}**: {item.get('description', '')}")

if st.session_state.plan_run and st.button("View Audit Trail"):
    st.json(st.session_state.plan_run.model_dump_json(indent=2))