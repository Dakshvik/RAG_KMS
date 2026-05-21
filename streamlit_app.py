import streamlit as st
import requests
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TGPA AI",
    page_icon="⚖️",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "token_ok" not in st.session_state:
    st.session_state.token_ok = False

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("⚖️ TGPA AI")

    st.markdown("### API Settings")

    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url
    )

    st.session_state.api_url = api_url

    token = st.text_input(
        "API Token",
        type="password"
    )

    # -----------------------------------------------------
    # CONNECT
    # -----------------------------------------------------
    if st.button("Connect"):

        if not token.strip():

            st.error("Enter API token")

        else:

            try:

                start = time.perf_counter()

                resp = requests.get(
                    f"{api_url}/health",
                    timeout=20
                )

                elapsed = round(
                    time.perf_counter() - start,
                    2
                )

                if resp.status_code == 200:

                    st.session_state.token = token

                    st.session_state.token_ok = True

                    st.success(
                        f"Connected ({elapsed}s)"
                    )

                    health = resp.json()

                    st.json(health)

                else:

                    st.error(
                        f"Server error: "
                        f"{resp.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to API server"
                )

            except Exception as e:

                st.error(str(e))

    # -----------------------------------------------------
    # WARMUP
    # -----------------------------------------------------
    if st.button("Warmup Pipeline"):

        try:

            headers = {
                "Authorization":
                f"Bearer {token}"
            }

            resp = requests.post(
                f"{api_url}/warmup",
                headers=headers,
                timeout=300
            )

            if resp.status_code == 200:

                st.success(
                    "Pipeline warmed up"
                )

            else:

                st.error(resp.text)

        except Exception as e:

            st.error(str(e))

    st.divider()

    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------
    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    st.divider()

    st.markdown("""
    ### Instructions

    1. Start Ollama
    2. Start Qdrant
    3. Start API server
    4. Connect
    5. Warmup
    6. Ask questions
    """)

# =========================================================
# MAIN AREA
# =========================================================
st.title("⚖️ TGPA AI Legal Assistant")

st.caption(
    "Hybrid RAG + Qdrant + Ollama"
)

# =========================================================
# DISPLAY CHAT
# =========================================================
for chat in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(chat["query"])

    with st.chat_message("assistant"):

        st.write(chat["answer"])

        if chat.get("route"):

            st.caption(
                f"Route: {chat['route']}"
            )

# =========================================================
# USER INPUT
# =========================================================
query = st.chat_input(
    "Ask a legal question..."
)

# =========================================================
# PROCESS QUERY
# =========================================================
if query:

    if not st.session_state.token_ok:

        st.warning(
            "Connect to API first"
        )

        st.stop()

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------
    with st.chat_message("user"):

        st.write(query)

    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------
    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.info(
            "Processing..."
        )

        try:

            headers = {

                "Authorization":
                f"Bearer {st.session_state.token}"
            }

            payload = {
                "query": query
            }

            start = time.perf_counter()

            response = requests.post(

                f"{st.session_state.api_url}/query",

                headers=headers,

                json=payload,

                timeout=300,
            )

            elapsed = round(
                time.perf_counter() - start,
                2
            )

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------
            if response.status_code == 200:

                data = response.json()

                result = data.get(
                    "result",
                    {}
                )

                answer = result.get(
                    "answer",
                    "No answer returned."
                )

                citations = result.get(
                    "citations",
                    []
                )

                timings = result.get(
                    "timings",
                    {}
                )

                route = result.get(
                    "route",
                    "unknown"
                )

                query_type = result.get(
                    "query_type",
                    "unknown"
                )

                thinking.empty()

                st.write(answer)

                st.caption(
                    f"Completed in {elapsed}s"
                )

                # ---------------------------------------------
                # ROUTE
                # ---------------------------------------------
                st.caption(
                    f"Route: {route}"
                )

                if route == "rag":

                    st.caption(
                        f"Query Type: "
                        f"{query_type}"
                    )

                # ---------------------------------------------
                # CITATIONS
                # ---------------------------------------------
                if citations:

                    with st.expander(
                        "📚 Citations"
                    ):

                        for i, cit in enumerate(
                            citations,
                            1
                        ):

                            st.markdown(
                                f"**{i}.** "
                                f"{cit.get('file')} "
                                f"(page {cit.get('page')})"
                            )

                            st.code(
                                cit.get(
                                    "text",
                                    ""
                                )[:500]
                            )

                # ---------------------------------------------
                # TIMINGS
                # ---------------------------------------------
                if timings:

                    with st.expander(
                        "⏱️ Timings"
                    ):

                        st.json(timings)

                # ---------------------------------------------
                # SAVE CHAT
                # ---------------------------------------------
                st.session_state.chat_history.append({

                    "query":
                    query,

                    "answer":
                    answer,

                    "route":
                    route
                })

            # -------------------------------------------------
            # ERROR
            # -------------------------------------------------
            else:

                thinking.empty()

                st.error(
                    f"API Error "
                    f"{response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            thinking.empty()

            st.error(
                "API server unreachable"
            )

        except requests.exceptions.Timeout:

            thinking.empty()

            st.error(
                "Request timed out"
            )

        except Exception as e:

            thinking.empty()

            st.error(str(e))