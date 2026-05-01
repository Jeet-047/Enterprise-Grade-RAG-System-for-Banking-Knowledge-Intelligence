import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Banking RAG Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= STYLING =========================
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }
        
        /* Title: keep emojis normal (gradient clip ruins emoji appearance) */
        .title-heading {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-bottom: 1rem;
        }
        .title-emojis {
            font-size: 3rem;
            line-height: 1;
            font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
        }
        .title-main-text {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .title-section {
            font-size: 2rem;
            font-weight: 700;
            color: #1f77e8;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }
        
        .subtitle-section {
            font-size: 1.3rem;
            font-weight: 600;
            color: #444;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .success-card {
            background: linear-gradient(135deg, #84fab015 0%, #8fd3f415 100%);
            border-left: 4px solid #84fab0;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        /* Response styling */
        .response-container {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Error styling */
        .error-container {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            color: #c62828;
        }
        
        /* Loading animation */
        .loading {
            display: inline-block;
        }
        
        /* Custom button styling */
        .button-container {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ========================= CONFIGURATION =========================
# Try to get API_BASE_URL from secrets, fallback to default
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
except (FileNotFoundError, KeyError):
    API_BASE_URL = "http://localhost:8000"

DEFAULT_TIMEOUT = 300

# Session state initialization
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "kb_token" not in st.session_state:
    st.session_state.kb_token = None
if "kb_token_expires" not in st.session_state:
    st.session_state.kb_token_expires = None
if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ========================= UTILITY FUNCTIONS =========================
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def format_json_output(data: Dict[str, Any]) -> str:
    """Format JSON data for display"""
    return json.dumps(data, indent=2)

def format_timestamp_for_display(ts: Any) -> str:
    """Convert timestamp values to a consistent local display format."""
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Handle epoch timestamps in seconds or milliseconds.
    if isinstance(ts, (int, float)):
        try:
            if ts > 1_000_000_000_000:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    # Handle string timestamps (ISO-8601 or already formatted strings).
    if isinstance(ts, str):
        ts_clean = ts.strip()
        if not ts_clean:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            iso_candidate = ts_clean.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(iso_candidate)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return ts_clean

    return str(ts)

def extract_timestamp(payload: Dict[str, Any]) -> Any:
    """Extract a timestamp field from known response keys."""
    for key in ("timestamp", "created_at", "generated_at", "time", "evaluated_at"):
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None

def stream_markdown_text(text: str, placeholder, delay_seconds: float = 0.008, chunk_size: int = 12):
    """Stream markdown text progressively while preserving final structure."""
    if not text:
        placeholder.markdown("")
        return
    rendered = []
    for i in range(0, len(text), chunk_size):
        rendered.append(text[i:i + chunk_size])
        placeholder.markdown("".join(rendered))
        time.sleep(delay_seconds)

def display_metric(label: str, value: str, icon: str = "📊"):
    """Display a metric in a card"""
    st.markdown(f"""
        <div class="metric-card">
            <h4>{icon} {label}</h4>
            <p style="font-size: 1.2rem; color: #667eea; font-weight: 600;">{value}</p>
        </div>
    """, unsafe_allow_html=True)

def display_error(error_msg: str):
    """Display error message"""
    st.markdown(f"""
        <div class="error-container">
            <strong>❌ Error:</strong> {error_msg}
        </div>
    """, unsafe_allow_html=True)

def display_success(success_msg: str):
    """Display success message"""
    st.markdown(f"""
        <div class="success-card">
            <strong>✅ Success:</strong> {success_msg}
        </div>
    """, unsafe_allow_html=True)

# ========================= SECTION 1: MAIN QUERY INTERFACE =========================
def render_query_section():
    """Render the main query/messaging section"""
    st.markdown('<h2 class="title-section">💬 Query & Intelligence Interface</h2>', unsafe_allow_html=True)
    
    api_health = check_api_health()
    if not api_health:
        display_error("API is not available. Please ensure the API server is running on " + API_BASE_URL)
        return
        
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.query_history = []
            st.rerun()
    
    # Existing chat history
    if not st.session_state.documents_indexed:
        st.warning("Upload and index at least one document from the sidebar before starting chat.")
    elif not st.session_state.chat_messages:
        st.info("Start chatting below. Ask anything about banking, policy, compliance, or risk.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.write(message["content"])
            if message["role"] == "assistant":
                metadata = []
                if message.get("source"):
                    metadata.append(f"Source: {message['source']}")
                if message.get("confidence") is not None:
                    metadata.append(f"Confidence: {message['confidence'] * 100:.1f}%")
                if message.get("latency"):
                    metadata.append(f"Latency: {message['latency']}")
                if metadata:
                    st.caption(" | ".join(metadata))

    # Query Input (chat style)
    query_input = st.chat_input(
        "Ask a question about banking, policies, compliance, etc.",
        disabled=not st.session_state.documents_indexed,
    )

    if query_input:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": query_input
        })

        with st.chat_message("user"):
            st.write(query_input)

        assistant_payload = {
            "role": "assistant",
            "content": "",
            "source": None,
            "confidence": None,
            "latency": None,
        }
        outcome = {"ok": False, "final_answer": None, "source": None, "confidence": None, "latency": None}

        with st.chat_message("assistant"):
            with st.spinner("🔄 Processing your query…"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"query": query_input},
                        timeout=DEFAULT_TIMEOUT
                    )

                    if response.status_code == 200:
                        result = response.json()
                        response_latency_seconds = response.elapsed.total_seconds()
                        response_latency_display = f"{response_latency_seconds:.3f}s"
                        final_answer = result.get("final_answer", "No answer available")
                        source = result.get("source", "Unknown")
                        confidence = result.get("confidence_score", 0)

                        st.session_state.query_history.append({
                            "query": query_input,
                            "latency": response_latency_display,
                            "result": result
                        })

                        outcome.update({
                            "ok": True,
                            "final_answer": final_answer,
                            "source": source,
                            "confidence": confidence,
                            "latency": response_latency_display,
                        })
                    else:
                        error_msg = f"API returned status code {response.status_code}"
                        display_error(error_msg)
                        assistant_payload["content"] = f"Sorry, I hit an error: {error_msg}"

                except requests.exceptions.Timeout:
                    error_msg = "Request timed out. Please try again."
                    display_error(error_msg)
                    assistant_payload["content"] = f"Sorry, I hit an error: {error_msg}"
                except requests.exceptions.ConnectionError:
                    error_msg = "Failed to connect to API. Please check if the server is running."
                    display_error(error_msg)
                    assistant_payload["content"] = f"Sorry, I hit an error: {error_msg}"
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    display_error(error_msg)
                    assistant_payload["content"] = f"Sorry, I hit an error: {error_msg}"

            if outcome["ok"]:
                response_placeholder = st.empty()
                stream_markdown_text(outcome["final_answer"], response_placeholder)
                conf = float(outcome["confidence"] if outcome["confidence"] is not None else 0.0)
                st.caption(
                    f"Source: {outcome['source']} | Confidence: {conf * 100:.1f}% | Latency: {outcome['latency']}"
                )
                assistant_payload.update({
                    "content": outcome["final_answer"],
                    "source": outcome["source"],
                    "confidence": outcome["confidence"],
                    "latency": outcome["latency"],
                })

            st.session_state.chat_messages.append(assistant_payload)


# ========================= SECTION 2: OTHER API ENDPOINTS =========================
def render_endpoints_section():
    """Render the other API endpoints section"""
    st.markdown('<h2 class="title-section">⚙️ Advanced API Endpoints</h2>', unsafe_allow_html=True)
    
    # Create tabs for different endpoints
    tab1, tab2, tab3 = st.tabs([
        "🔐 Knowledge Base Access",
        "🔍 Debug Query Analysis",
        "📊 Raw API Response"
    ])
    
    # ==================== TAB 1: KB ACCESS ====================
    with tab1:
        st.markdown('<p class="subtitle-section">Knowledge Base Token & Fetch</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # --- KB Token Generation ---
        with col1:
            st.markdown("#### 🔐 Generate KB Token")
            st.write("Generate a secure token for KB access")
            
            api_key_input = st.text_input(
                "Enter your API Key",
                type="password",
                placeholder="Your internal API key",
                key="api_key_input"
            )
            
            if st.button("🔓 Generate Token", key="gen_token", use_container_width=True):
                if not api_key_input:
                    display_error("Please enter an API key")
                else:
                    with st.spinner("Generating token..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/kb/token",
                                headers={"X-API-KEY": api_key_input},
                                timeout=DEFAULT_TIMEOUT
                            )
                            
                            if response.status_code == 200:
                                token_data = response.json()
                                st.session_state.kb_token = token_data.get('token')
                                st.session_state.kb_token_expires = token_data.get('expires_in')
                                
                                display_success("Token generated successfully!")
                                
                                st.json({
                                    "token": token_data.get('token'),
                                    "expires_in": token_data.get('expires_in'),
                                    "generated_at": datetime.now().isoformat()
                                })
                            else:
                                display_error(f"Failed to generate token: {response.status_code}")
                                
                        except Exception as e:
                            display_error(f"Error: {str(e)}")
        
        # --- KB Data Fetch ---
        with col2:
            st.markdown("#### 📚 Fetch KB Data")
            st.write("Retrieve data from Knowledge Base")
            
            if st.session_state.kb_token is None:
                st.warning("⚠️ Please generate a token first using the panel on the left")
            else:
                display_success(f"Token active (expires in {st.session_state.kb_token_expires}s)")
                
                kb_query = st.text_input(
                    "Enter KB Query",
                    placeholder="What data do you want to fetch?",
                    key="kb_query"
                )
                
                if st.button("📥 Fetch Data", key="fetch_kb", use_container_width=True):
                    if not kb_query:
                        display_error("Please enter a query")
                    else:
                        with st.spinner("Fetching KB data..."):
                            try:
                                response = requests.post(
                                    f"{API_BASE_URL}/kb/fetch",
                                    params={"query": kb_query},
                                    headers={
                                        "Authorization": f"Bearer {st.session_state.kb_token}"
                                    },
                                    timeout=DEFAULT_TIMEOUT
                                )
                                
                                if response.status_code == 200:
                                    kb_data = response.json()
                                    
                                    display_success("Data fetched successfully!")
                                    
                                    if kb_data.get("data"):
                                        with st.expander("📋 KB Data Details", expanded=True):
                                            st.json(kb_data)
                                    else:
                                        st.info(kb_data.get("message", "No data available"))
                                else:
                                    display_error(f"Failed to fetch data: {response.status_code}")
                                    
                            except Exception as e:
                                display_error(f"Error: {str(e)}")
    
    # ==================== TAB 2: DEBUG ANALYSIS ====================
    with tab2:
        st.markdown('<p class="subtitle-section">Debug Query Analysis</p>', unsafe_allow_html=True)
        st.write("Analyze queries in detail with hallucination detection and debug information")
        
        debug_query = st.text_area(
            "Enter Query for Debug Analysis",
            placeholder="Enter a query to analyze...",
            height=100,
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Analyze Query", use_container_width=True, key="analyze_debug"):
            if not debug_query:
                display_error("Please enter a query")
            else:
                with st.spinner("🔍 Analyzing query..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/query/debug",
                            json={"query": debug_query},
                            timeout=DEFAULT_TIMEOUT
                        )
                        
                        if response.status_code == 200:
                            debug_result = response.json()
                            
                            display_success("Analysis completed!")
                            
                            # Display Results in Columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                display_metric(
                                    "Similarity Score",
                                    f"{debug_result.get('similarity_score', 0):.4f}",
                                    "📊"
                                )
                            with col2:
                                hallucination = debug_result.get('hallucination_decision', False)
                                status = "🚨 Detected" if hallucination else "✅ Not Detected"
                                display_metric(
                                    "Hallucination Status",
                                    status,
                                    "👁️"
                                )
                            
                            # Retrieved Chunks
                            if debug_result.get('retrieved_chunks'):
                                with st.expander("📚 Retrieved Chunks", expanded=True):
                                    for idx, chunk in enumerate(debug_result.get('retrieved_chunks', []), 1):
                                        st.write(f"**Chunk {idx}:**")
                                        st.write(chunk)
                                        st.divider()
                            
                            # KB Data (if hallucination detected)
                            if debug_result.get('kb_used'):
                                st.divider()
                                st.markdown("#### 🛡️ Hallucination Mitigation")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**KB Data Retrieved:**")
                                    with st.expander("View KB Data"):
                                        st.json(debug_result.get('kb_data'))
                                
                                with col2:
                                    st.markdown("**KB-based Answer:**")
                                    st.info(debug_result.get('kb_answer', 'No KB answer available'))
                            
                            # Full Response
                            with st.expander("📋 Full Response JSON"):
                                st.json(debug_result)
                        else:
                            display_error(f"API returned status code {response.status_code}")
                            
                    except Exception as e:
                        display_error(f"Error: {str(e)}")
    
    # ==================== TAB 3: RAW API RESPONSE ====================
    with tab3:
        st.markdown('<p class="subtitle-section">Raw API Response Explorer</p>', unsafe_allow_html=True)

        endpoint_config = {
            "/retrieval/logs": {
                "method": "GET",
                "description": "Fetch the latest retrieval log file",
            },
            "/chunks/inspect": {
                "method": "GET",
                "description": "Inspect generated chunks from preprocessing",
            },
            "/evaluate": {
                "method": "POST",
                "description": "Evaluate multiple queries against the RAG pipeline",
            },
            "/health": {
                "method": "GET",
                "description": "Check API health status",
            },
        }

        endpoint = st.selectbox("Select Endpoint", list(endpoint_config.keys()))
        method_choice = endpoint_config[endpoint]["method"]

        st.info(f"Method: {method_choice} | {endpoint_config[endpoint]['description']}")

        payload = None
        if endpoint == "/evaluate":
            st.write("**Evaluation Queries (one query per line):**")
            raw_queries = st.text_area(
                "Test Queries",
                value="What is KYC?\nWhat are AML compliance requirements?",
                height=120,
                label_visibility="collapsed",
            )
            test_queries = [line.strip() for line in raw_queries.splitlines() if line.strip()]
            payload = {"test_queries": test_queries}

        # Headers
        with st.expander("Advanced Options", expanded=False):
            custom_headers = st.text_area(
                "Custom Headers (JSON)",
                value='{"Content-Type": "application/json"}',
                height=100,
            )
            token_for_kb = st.text_input("Bearer Token (for KB endpoints):", type="password")

        if st.button("🚀 Send Request", use_container_width=True, key="send_raw"):
            with st.spinner("Sending request..."):
                try:
                    url = f"{API_BASE_URL}{endpoint}"

                    headers = {"Content-Type": "application/json"}
                    try:
                        headers.update(json.loads(custom_headers))
                    except:
                        display_error("Invalid custom headers JSON. Using default headers only.")

                    if token_for_kb:
                        headers["Authorization"] = f"Bearer {token_for_kb}"

                    if endpoint == "/evaluate":
                        if not payload or not payload.get("test_queries"):
                            display_error("Please enter at least one test query for evaluation.")
                            return
                        response = requests.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=DEFAULT_TIMEOUT,
                        )
                    else:
                        response = requests.get(
                            url,
                            headers=headers,
                            timeout=DEFAULT_TIMEOUT,
                        )

                    st.divider()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status_color = "🟢" if response.status_code < 400 else "🔴"
                        display_metric("Status Code", f"{status_color} {response.status_code}", "📊")
                    with col2:
                        display_metric("Content Type", response.headers.get("content-type", "Unknown"), "📄")
                    with col3:
                        display_metric("Response Time", f"{response.elapsed.total_seconds():.3f}s", "⏱️")
                    response_received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"Response received at: {response_received_at}")

                    st.markdown('<p class="subtitle-section">Response Body</p>', unsafe_allow_html=True)

                    parsed_json = None
                    try:
                        parsed_json = response.json()
                    except:
                        parsed_json = None

                    if parsed_json is None:
                        st.code(response.text)
                    elif endpoint == "/health":
                        health_status = parsed_json.get("status", "unknown")
                        health_icon = "✅" if str(health_status).lower() == "ok" else "⚠️"
                        st.success(f"{health_icon} API Health: {health_status}")
                        extracted_ts = extract_timestamp(parsed_json)
                        if extracted_ts is not None:
                            st.write(f"**Timestamp:** {format_timestamp_for_display(extracted_ts)}")
                        with st.expander("Full JSON"):
                            st.json(parsed_json)
                    elif endpoint == "/retrieval/logs":
                        st.write(f"**Log File:** `{parsed_json.get('file', 'N/A')}`")
                        log_content = parsed_json.get("content", "")
                        if log_content:
                            st.text_area("Latest Log Content", value=log_content, height=300, disabled=True)
                        else:
                            st.info(parsed_json.get("message", "No log content available"))
                        with st.expander("Full JSON"):
                            st.json(parsed_json)
                    elif endpoint == "/chunks/inspect":
                        chunks = parsed_json.get("chunks", [])
                        if chunks:
                            st.write(f"**Total Chunks:** {len(chunks)}")
                            st.dataframe(chunks, width="stretch")
                        else:
                            st.info("No chunks available.")
                        with st.expander("Full JSON"):
                            st.json(parsed_json)
                    elif endpoint == "/evaluate":
                        st.write(f"**Total Queries:** {parsed_json.get('total_queries', 0)}")
                        st.write(f"**Average Confidence:** {parsed_json.get('average_confidence', 0.0):.4f}")
                        extracted_ts = extract_timestamp(parsed_json)
                        if extracted_ts is not None:
                            st.write(f"**Timestamp:** {format_timestamp_for_display(extracted_ts)}")
                        results = parsed_json.get("results", [])
                        if results:
                            st.dataframe(results, width="stretch")
                        else:
                            st.info("No evaluation results returned.")
                        with st.expander("Full JSON"):
                            st.json(parsed_json)
                    else:
                        st.json(parsed_json)

                except requests.exceptions.Timeout:
                    display_error("Request timed out")
                except requests.exceptions.ConnectionError:
                    display_error("Failed to connect to API")
                except Exception as e:
                    display_error(f"Error: {str(e)}")


# ========================= MAIN APP =========================
def main():
    # Header
    st.markdown(
        '''
        <h1 class="title-heading">
            <span class="title-emojis">🏦</span>
            <span class="title-main-text">Banking RAG Intelligence System</span>
        </h1>
        ''',
        unsafe_allow_html=True,
    )
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    border-left: 4px solid #667eea; 
                    padding: 1.5rem; 
                    border-radius: 0.5rem; 
                    margin-bottom: 2rem;">
            <p style="margin: 0;">
                <strong>Welcome to the Advanced RAG System!</strong> This interface provides intelligent querying, 
                knowledge base access, and advanced debugging capabilities for your banking knowledge intelligence platform.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        nav_selection = st.radio(
            "Select Interface",
            ["💬 Query Interface", "⚙️ API Endpoints"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### 🔌 API Status")
        api_health = check_api_health()
        status_color = "🟢" if api_health else "🔴"
        st.info(f"{status_color} **{'Connected' if api_health else 'Disconnected'}**")
        
        st.divider()
        
        st.markdown("### 📄 Document Upload")
        st.caption("Files: PDF, TXT, Word (.doc / .docx), CSV · URLs use WebBaseLoader (one https URL per line).")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "doc", "docx", "csv"],
            accept_multiple_files=True,
        )
        web_urls_input = st.text_area(
            "Or index web pages (optional)",
            height=72,
            placeholder="https://example.com/page-one\nhttps://example.com/policy",
            label_visibility="visible",
            help="Each line must start with http:// or https://. Indexed together with uploaded files.",
        )
        if st.button("📑 Index Document", use_container_width=True):
            urls_text = (web_urls_input or "").strip()
            if uploaded_files or urls_text:
                with st.spinner("Chunking and indexing documents..."):
                    try:
                        req_kwargs = {"timeout": DEFAULT_TIMEOUT}
                        if uploaded_files:
                            req_kwargs["files"] = [
                                ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
                                for file in uploaded_files
                            ]
                        if urls_text:
                            req_kwargs["data"] = {"web_urls": urls_text}
                        response = requests.post(
                            f"{API_BASE_URL}/documents/index",
                            **req_kwargs,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            indexed_count = int(result.get("indexed_documents", 0))
                            st.session_state.documents_indexed = indexed_count > 0
                            st.session_state.indexed_files = result.get("uploaded_files", [])
                            if st.session_state.documents_indexed:
                                st.success(
                                    f"Successfully chunked and indexed {indexed_count} document(s)! You can now start chatting."
                                )
                            else:
                                st.warning(result.get("message", "No supported files were indexed."))
                        else:
                            st.session_state.documents_indexed = False
                            display_error(f"Indexing failed with status code {response.status_code}")
                    except requests.exceptions.Timeout:
                        st.session_state.documents_indexed = False
                        display_error("Indexing request timed out. Please try again.")
                    except requests.exceptions.ConnectionError:
                        st.session_state.documents_indexed = False
                        display_error("Failed to connect to API. Ensure FastAPI is running.")
                    except Exception as e:
                        st.session_state.documents_indexed = False
                        display_error(f"Failed to index documents: {str(e)}")
            else:
                st.warning("Upload at least one file or enter at least one https:// URL, then click Index.")
        if st.session_state.indexed_files:
            st.caption("Indexed files: " + ", ".join(st.session_state.indexed_files))
        
        st.divider()
        
        st.markdown("### ℹ️ About")
        st.info("""
            **Enterprise-Grade RAG System**
            
            Version: 1.0.0
            
            This system provides:
            - 🎯 Intelligent query processing
            - 🔐 Secure KB access
            - 🛡️ Hallucination detection
            - 📊 Advanced debugging
            - 📄 Document chunking & indexing
            
            [Documentation](https://github.com/Jeet-047/Enterprise-Grade-RAG-System-for-Banking-Knowledge-Intelligence)
        """)
    
    # Main Content - Based on Navigation Selection
    if nav_selection == "💬 Query Interface":
        render_query_section()
    elif nav_selection == "⚙️ API Endpoints":
        render_endpoints_section()
    
    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">
            <p>🏦 Banking RAG Intelligence System © 2024 | Powered by FastAPI + Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
