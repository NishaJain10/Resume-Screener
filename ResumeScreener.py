import os
import hashlib
from typing import List
import chromadb
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()


class CandidateSummary(BaseModel):
    name: str = Field(description="Candidate's full name")
    years_of_experience: int = Field(description="Total years of professional experience")
    top_skills: List[str] = Field(description="List of core technical skills explicitly mentioned")
    current_role: str = Field(description="The candidate's most recent job title")

# Initialize the Azure OpenAI model
@st.cache_resource
def get_llm():
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    return llm, llm.with_structured_output(CandidateSummary)

llm, structured_llm = get_llm()


@st.cache_resource #streamlit decorator
def get_vector_store():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(
        name="resumes",
        metadata={"hnsw:space": "cosine"}, #cosine similarity that it will check with jd
    )


st.set_page_config(page_title="Local AI Resume Ranker", layout="wide")
st.title("🎯 AI Resume Screener & Ranker")
st.caption("Powered by Azure OpenAI & LangChain")

st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("📋 Input Criteria")
    
    folder_path = st.text_input(
        "Enter the absolute path to your local resume folder:",
        value="./resumes",
        help="Make sure this folder exists on your system and contains PDF resumes."
    )
    
    job_description = st.text_area(
        "Paste the Job Description:",
        height=250,
        placeholder="e.g., Looking for a Senior Backend Developer with 5+ years of experience in Python and AWS..."
    )
    
    start_ranking = st.button("🚀 Analyze and Rank Resumes", type="primary", use_container_width=True)

with col2:
    st.header("🏆 Ranked Results")
    
    if start_ranking:
        st.session_state.pop("poc1app_results", None)

        if not os.path.exists(folder_path):
            st.error(f"❌ The directory `{folder_path}` does not exist on your system. Please check the path.")
        elif not job_description.strip():
            st.warning("⚠️ Please provide a job description to match against.")
        else:
            collection = get_vector_store()

            pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

            if not pdf_files:
                st.warning(f"No PDF files found in `{folder_path}`.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

              
                for index, filename in enumerate(pdf_files):
                    status_text.text(f"Indexing file {index + 1}/{len(pdf_files)}: {filename}...")
                    file_path = os.path.join(folder_path, filename)

                    try:
                        # Unique id from file contents so we never re-embed the same resume
                        with open(file_path, "rb") as f:
                            doc_id = hashlib.md5(f.read()).hexdigest()

                        # Skip files already stored in the vector DB (saves LLM calls)
                        if collection.get(ids=[doc_id])["ids"]:
                            progress_bar.progress((index + 1) / len(pdf_files))
                            continue

                        # Load and combine text from the system PDF
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        full_text = " ".join([doc.page_content for doc in docs])

                        # Extract structured data
                        extraction_prompt = ChatPromptTemplate.from_template(
                            "Extract the candidate profile accurately. Return ONLY the requested fields.\n\nText:\n{text}"
                        )
                        chain = extraction_prompt | structured_llm
                        profile = chain.invoke({"text": full_text}).model_dump()

                        # Store the resume text (embedded) + profile metadata in the vector DB
                        collection.add(
                            ids=[doc_id],
                            documents=[full_text],
                            metadatas=[{
                                "name": profile["name"],
                                "years_of_experience": profile["years_of_experience"],
                                "current_role": profile["current_role"],
                                "top_skills": ", ".join(profile["top_skills"]),
                                "filename": filename,
                            }],
                        )
                    except Exception as e:
                        st.error(f"Error reading {filename}: {str(e)}")

                    # Update progress
                    progress_bar.progress((index + 1) / len(pdf_files))

                status_text.text("Running semantic search against the job description...")

                
                total = collection.count()
                if total == 0:
                    st.warning("No candidates in the vector DB yet.")
                else:
                    # Retrieve resumes most semantically similar to the JD
                    results = collection.query(
                        query_texts=[job_description],
                        n_results=total,
                    )
                    metadatas = results["metadatas"][0]
                    distances = results["distances"][0]

                    
                    formatted_candidates = ""
                    comparison_rows = []
                    for rank, (meta, dist) in enumerate(zip(metadatas, distances), 1):
                        similarity = max(0.0, min(1.0, 1 - float(dist)))
                        comparison_rows.append({
                            "Semantic Rank": rank,
                            "Candidate": meta["name"],
                            "Experience (Years)": meta["years_of_experience"],
                            "Current Role": meta["current_role"],
                            "Top Skills": meta["top_skills"],
                            "Resume": meta["filename"],
                            "Match Score": similarity,
                        })
                        formatted_candidates += (
                            f"\nCandidate {rank}:\n"
                            f"- Name: {meta['name']}\n"
                            f"- Exp: {meta['years_of_experience']} years\n"
                            f"- Role: {meta['current_role']}\n"
                            f"- Skills: {meta['top_skills']}\n"
                            f"- Vector similarity to JD: {similarity:.3f}\n"
                        )

                    ranking_prompt = ChatPromptTemplate.from_template(
                        """You are an expert recruiter. The candidates below were retrieved from a vector database, pre-sorted by semantic similarity to the Job Description.
                        Re-rank them from best to worst based on experience and skills, and provide a detailed 1-sentence reason for each.

                        Job Description: {jd}
                        Candidates: {candidates}

                        Output Format:
                        1. [Name] - [Reason]
                        2. [Name] - [Reason]
                        """
                    )
                    ranking_chain = ranking_prompt | llm

                    with st.spinner("Azure OpenAI is evaluating the shortlist..."):
                        final_report = ranking_chain.invoke({
                            "jd": job_description,
                            "candidates": formatted_candidates,
                        })

                    # Clear loading elements and display final output
                    status_text.empty()
                    progress_bar.empty()

                    st.session_state["poc1app_results"] = {
                        "report": final_report.content,
                        "comparison_rows": comparison_rows,
                        "resume_folder": os.path.realpath(folder_path),
                    }

    saved_results = st.session_state.get("poc1app_results")
    if saved_results:
        comparison_rows = saved_results["comparison_rows"]

        st.success("Analysis Complete!")
        st.markdown(saved_results["report"])

        st.subheader("📊 Candidate Comparison")
        comparison_event = st.dataframe(
            comparison_rows,
            column_config={
                "Semantic Rank": st.column_config.NumberColumn(
                    "Rank",
                    format="%d",
                ),
                "Candidate": st.column_config.TextColumn(
                    "Candidate",
                    width="medium",
                ),
                "Experience (Years)": st.column_config.NumberColumn(
                    "Experience",
                    format="%d years",
                ),
                "Current Role": st.column_config.TextColumn(
                    "Current Role",
                    width="medium",
                ),
                "Top Skills": st.column_config.TextColumn(
                    "Top Skills",
                    width="large",
                ),
                "Resume": st.column_config.TextColumn(
                    "Resume File",
                    width="medium",
                ),
                "Match Score": st.column_config.ProgressColumn(
                    "Semantic Match",
                    help="Similarity between the resume and job description",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            },
            column_order=[
                "Semantic Rank",
                "Candidate",
                "Experience (Years)",
                "Current Role",
                "Top Skills",
                "Resume",
                "Match Score",
            ],
            hide_index=True,
            use_container_width=True,
            key="candidate_comparison",
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = comparison_event.selection.rows
        if selected_rows:
            selected_candidate = comparison_rows[selected_rows[0]]
            resume_name = selected_candidate["Resume"]
            resume_folder = saved_results["resume_folder"]
            resume_path = os.path.realpath(os.path.join(resume_folder, resume_name))

            if os.path.basename(resume_name) != resume_name:
                st.error("The selected resume path is outside the configured resume folder.")
            elif os.path.commonpath([resume_folder, resume_path]) != resume_folder:
                st.error("The selected resume path is outside the configured resume folder.")
            elif not os.path.isfile(resume_path):
                st.error(
                    f"Resume file `{resume_name}` is no longer available "
                    "in the configured folder."
                )
            else:
                with open(resume_path, "rb") as resume_file:
                    st.download_button(
                        label=f"⬇️ Download {selected_candidate['Candidate']}'s Resume",
                        data=resume_file.read(),
                        file_name=resume_name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
        else:
            st.caption("Select a candidate row to download their resume.")

        # Show the raw vector matches behind the scenes
        with st.expander("🔍 Vector DB matches (semantic similarity)"):
            st.json(comparison_rows)
    else:
        st.info("Fill out the inputs on the left and click 'Analyze and Rank' to see the magic happen.")