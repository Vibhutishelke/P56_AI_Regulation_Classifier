import io, json, re
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
from gtts import gTTS
from pypdf import PdfReader

st.set_page_config(page_title="P56 | AI Regulation Classifier", page_icon="⚖️", layout="wide")

EU_URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj"
EU_GUIDE = "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai"
DPDP_URL = "https://www.indiacode.nic.in/indiacode/handle/123456789/22037?view_type=browse"
RULES_URL = "https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf"

# Short retrieval aids. Upload the full official laws/rules in the sidebar for deeper retrieval.
DOCS = [
 {"id":"EU-A5","jurisdiction":"EU AI Act","reference":"Article 5","title":"Prohibited AI practices","text":"Article 5 prohibits specified AI practices including certain harmful manipulation or exploitation, social scoring, and certain biometric and emotion-recognition practices. Exact facts, context and statutory exceptions matter.","url":EU_URL},
 {"id":"EU-A6","jurisdiction":"EU AI Act","reference":"Article 6","title":"High-risk classification rules","text":"Article 6 sets rules for high-risk AI systems, including systems covered by product-safety legislation and certain Annex III use cases, subject to conditions and exceptions.","url":EU_URL},
 {"id":"EU-ANNEX-III","jurisdiction":"EU AI Act","reference":"Annex III","title":"High-risk use-case areas","text":"Annex III lists specified high-risk areas including biometrics, critical infrastructure, education, employment, essential services, law enforcement, migration and border control, and justice and democratic processes.","url":EU_URL},
 {"id":"EU-A50","jurisdiction":"EU AI Act","reference":"Article 50","title":"Transparency duties","text":"Article 50 establishes transparency duties for certain AI systems, including informing people when they interact directly with AI and disclosure or marking for certain synthetic or manipulated content, subject to conditions.","url":EU_URL},
 {"id":"EU-A9","jurisdiction":"EU AI Act","reference":"Article 9","title":"Risk management","text":"Article 9 requires a risk management system for high-risk AI systems across the lifecycle to identify, estimate, evaluate and mitigate risks and test the system.","url":EU_URL},
 {"id":"EU-A10","jurisdiction":"EU AI Act","reference":"Article 10","title":"Data governance","text":"Article 10 sets data and data-governance requirements for high-risk AI systems, including relevant data quality practices for training, validation and testing datasets where applicable.","url":EU_URL},
 {"id":"DPDP-S3","jurisdiction":"India DPDP Act 2023","reference":"Section 3","title":"Application","text":"Section 3 describes the Act's application to digital personal data processed in India and certain processing outside India connected with offering goods or services to Data Principals in India, subject to statutory exclusions.","url":DPDP_URL},
 {"id":"DPDP-S4","jurisdiction":"India DPDP Act 2023","reference":"Section 4","title":"Grounds for processing","text":"Section 4 allows processing of personal data only in accordance with the Act for a lawful purpose, based on consent or certain legitimate uses specified in the Act.","url":DPDP_URL},
 {"id":"DPDP-S5","jurisdiction":"India DPDP Act 2023","reference":"Section 5","title":"Notice","text":"Section 5 requires a notice to the Data Principal describing the personal data and purpose of processing and explaining how rights and complaints may be exercised, subject to the Act and Rules.","url":DPDP_URL},
 {"id":"DPDP-S6","jurisdiction":"India DPDP Act 2023","reference":"Section 6","title":"Consent","text":"Section 6 addresses consent, including free, specific, informed, unconditional and unambiguous consent through clear affirmative action, and withdrawal, subject to the Act.","url":DPDP_URL},
 {"id":"DPDP-S8","jurisdiction":"India DPDP Act 2023","reference":"Section 8","title":"Data Fiduciary obligations","text":"Section 8 sets general Data Fiduciary obligations including responsibility for compliance, security safeguards, breach-related duties, accuracy in certain circumstances and erasure when retention is no longer necessary, subject to the Act.","url":DPDP_URL},
 {"id":"DPDP-S9","jurisdiction":"India DPDP Act 2023","reference":"Section 9","title":"Children's data","text":"Section 9 addresses children's personal data, including verifiable parental consent and restrictions on specified processing, subject to statutory exceptions and prescribed Rules.","url":DPDP_URL},
 {"id":"DPDP-RULES","jurisdiction":"India DPDP Rules 2025","reference":"Rules and commencement notification","title":"Phased commencement","text":"The DPDP Rules 2025 and Act commencement notifications use phased commencement. Assessments must check the effective date of each provision rather than assume every provision is already in force.","url":RULES_URL},
]

CASES = [
("AI CV screening","Ranks job applicants and filters CVs before recruiter review.","HIGH – likely employment Annex III","Personal data: assess notice, lawful basis, purpose, security, retention and rights."),
("Student emotion detection","Uses facial/emotion analysis to infer student emotions during exams.","PROHIBITED – likely workplace/education emotion-recognition ban; verify narrow exceptions","Student data: assess lawful basis, notice, safeguards and child-specific requirements."),
("Customer-support chatbot","Answers customer questions and tells users they are interacting with AI.","LIMITED / TRANSPARENCY – Article 50 may apply","If personal data is processed, assess Sections 4–6 and 8."),
("Spam email filter","Moves likely spam emails into a junk folder for a private user.","MINIMAL – context-dependent","Email may contain personal data; assess purpose, lawful basis, security and retention."),
("Medical diagnosis support","Analyses scans and suggests diagnoses for a clinician to review.","HIGH – may fall under regulated medical-device product rules","Health data: assess lawful basis, safeguards, access and retention."),
("Public-authority social scoring","Scores people across unrelated contexts and causes detrimental treatment.","PROHIBITED – if Article 5 social-scoring conditions are met","Separate DPDP assessment still required for personal data."),
("Factory predictive maintenance","Uses machine sensor data to predict equipment maintenance; no people are evaluated.","MINIMAL – unless safety-component/high-risk context applies","Likely outside DPDP if no personal data; check worker monitoring data."),
("Personal-loan credit scoring","Scores a natural person's creditworthiness for a loan decision.","HIGH – Annex III creditworthiness assessment, subject to exclusions","Assess personal data, notice, lawful basis, security, retention and rights."),
("Synthetic marketing image generator","Creates fictional product images without identifiable people.","MINIMAL – transparency duties depend on content/use","DPDP may not apply if no personal data is processed."),
("College facial attendance","Uses face templates to identify students and record attendance.","CONTEXT-DEPENDENT – biometric rules and any prohibited purpose must be checked","Biometric data is personal data; assess notice, lawful basis, security, retention and children.")
]

@st.cache_resource(show_spinner="Loading Sentence-BERT model (first run may take a few minutes)...")
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def make_index(documents, model):
    texts = [f"{d['jurisdiction']} {d['reference']} {d['title']} {d['text']}" for d in documents]
    vectors = model.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve(query, documents, model, index, k=7):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, min(k, len(documents)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(documents[int(idx)])
            item["similarity"] = float(score)
            results.append(item)
    return results

def read_upload(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.getvalue().decode("utf-8", errors="ignore")

def gemini(api_key, model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {"contents":[{"parts":[{"text":prompt}]}],
               "generationConfig":{"temperature":0.1,"responseMimeType":"application/json"}}
    response = requests.post(url, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

def build_prompt(case_text, evidence):
    evidence_text = "\n\n".join(
        f"[{e['id']}] {e['jurisdiction']} | {e['reference']} | {e['title']}\n{e['text']}\nURL: {e['url']}"
        for e in evidence
    )
    return f"""
You are a careful educational regulatory screening assistant.
Assess this AI system separately under the EU AI Act and India's DPDP Act 2023 and relevant Rules/commencement notifications.
Do not give legal advice. Do not invent legal rules or citations. The retrieved material may be incomplete.
EU category must be one of: PROHIBITED/UNACCEPTABLE, HIGH, LIMITED/TRANSPARENCY, MINIMAL, or INSUFFICIENT INFORMATION.
Explain the reason and mention relevant article/annex and any condition or exception. If uncertain, say so.
For DPDP, explain whether digital personal data appears to be involved and list potentially relevant sections, data-handling gaps, missing facts and practical next steps. Do not mark compliance as PASS just because facts are missing.
Every finding must include evidence_ids chosen only from the IDs below. If no evidence supports a finding, use an empty array and say evidence is insufficient.
Return valid JSON with keys:
eu_category, eu_reasoning, eu_evidence_ids (array), eu_obligations (array of objects with finding and evidence_ids),
dpdp_assessment, dpdp_findings (array of objects with finding and evidence_ids),
dpdp_gaps (array), missing_information (array), recommended_actions (array), confidence, disclaimer.
Use plain language suitable for a college demonstration.
USE CASE:
{case_text}
RETRIEVED SOURCES:
{evidence_text}
"""

def show_citations(ids, evidence_by_id):
    valid = [evidence_by_id[x] for x in ids if x in evidence_by_id]
    if not valid:
        st.caption("No retrieved source directly supports this point. Treat it as an unverified observation.")
    for e in valid:
        st.markdown(f"- **[{e['id']}] {e['jurisdiction']} — {e['reference']}: {e['title']}** · [Official source]({e['url']})")

st.title("⚖️ P56 — AI Regulation Use-Case Classifier")
st.caption("EU AI Act risk screening + India DPDP Act 2023 data-handling review | Educational prototype")

with st.sidebar:
    st.header("Settings")
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = st.text_input("Gemini API key", value=secret_key, type="password")
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash")
    st.caption("Never share your API key or commit it to GitHub.")
    uploads = st.file_uploader("Optional: upload official regulation PDFs/TXT", type=["pdf","txt"], accept_multiple_files=True)
    st.caption("Uploaded source files are used only for this session. Use official consolidated text for best results.")

documents = [dict(x) for x in DOCS]
for upload in uploads or []:
    try:
        raw = read_upload(upload)
        chunks = [raw[i:i+1600] for i in range(0, len(raw), 1300) if raw[i:i+1600].strip()]
        for n, chunk in enumerate(chunks, 1):
            documents.append({"id":f"FILE-{re.sub(r'[^A-Za-z0-9]','',upload.name)[:12]}-{n}",
              "jurisdiction":"Uploaded source","reference":f"{upload.name}, chunk {n}",
              "title":"Uploaded regulation text","text":chunk,"url":"See uploaded official file"})
    except Exception as exc:
        st.warning(f"Could not read {upload.name}: {exc}")

with st.spinner("Preparing semantic search..."):
    model = get_model()
    index = make_index(documents, model)

tab_classify, tab_cases, tab_retrieve, tab_about = st.tabs(
    ["Classify a use case", "10 sample cases", "Find regulation text", "How the project works"]
)

with tab_classify:
    st.subheader("Enter your AI use case")
    with st.form("case_form"):
        name = st.text_input("AI system name", placeholder="Example: AI recruitment screening")
        purpose = st.text_area("Purpose and intended use")
        people = st.text_input("Who is affected?", placeholder="Applicants, students, customers...")
        decisions = st.text_area("What decisions or recommendations does it make?")
        data = st.text_area("What data is collected?", placeholder="Personal data, biometrics, health, financial, children's data...")
        oversight = st.text_area("Human oversight")
        handling = st.text_area("Data handling", placeholder="Notice, consent/lawful basis, sharing, storage, security, retention, deletion...")
        submitted = st.form_submit_button("Classify and explain", type="primary")
    if submitted:
        if not name.strip() or not purpose.strip():
            st.error("Please enter at least the system name and purpose.")
        elif not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        else:
            case_text = f"Name: {name}\nPurpose: {purpose}\nAffected people: {people}\nDecisions: {decisions}\nData: {data}\nHuman oversight: {oversight}\nData handling: {handling}"
            with st.spinner("Retrieving legal passages and asking Gemini..."):
                evidence = retrieve(case_text, documents, model, index)
                try:
                    answer = json.loads(gemini(api_key, model_name, build_prompt(case_text, evidence)))
                    st.session_state["p56_answer"] = answer
                    st.session_state["p56_evidence"] = evidence
                    st.session_state["p56_case"] = case_text
                except Exception as exc:
                    st.error(f"Could not complete the LLM request: {exc}")

    if "p56_answer" in st.session_state:
        answer = st.session_state["p56_answer"]
        evidence = st.session_state["p56_evidence"]
        evidence_by_id = {e["id"]:e for e in evidence}
        st.divider()
        st.subheader("Assessment result")
        left, right = st.columns(2)
        with left:
            st.markdown("### EU AI Act")
            st.info(answer.get("eu_category", "Insufficient information"))
            st.write(answer.get("eu_reasoning", ""))
            show_citations(answer.get("eu_evidence_ids", []), evidence_by_id)
            for item in answer.get("eu_obligations", []):
                st.write("•", item.get("finding", ""))
                show_citations(item.get("evidence_ids", []), evidence_by_id)
        with right:
            st.markdown("### India DPDP Act")
            st.write(answer.get("dpdp_assessment", ""))
            for item in answer.get("dpdp_findings", []):
                st.write("•", item.get("finding", ""))
                show_citations(item.get("evidence_ids", []), evidence_by_id)
            for gap in answer.get("dpdp_gaps", []):
                st.warning(gap)
        st.markdown("### Missing information")
        for item in answer.get("missing_information", []): st.write("•", item)
        st.markdown("### Recommended actions")
        for item in answer.get("recommended_actions", []): st.write("•", item)
        st.caption(f"Confidence: {answer.get('confidence','not stated')} · {answer.get('disclaimer','Educational screening only; obtain qualified review.')}")
        with st.expander("All retrieved evidence"):
            for e in evidence:
                st.markdown(f"**[{e['id']}] {e['jurisdiction']} — {e['reference']} — {e['title']}**")
                st.write(e["text"])
                st.markdown(f"[Open source]({e['url']}) · Similarity: {e['similarity']:.3f}")
        report = {"case":st.session_state["p56_case"],"result":answer,"retrieved_evidence":evidence,
                  "generated_at":datetime.now(timezone.utc).isoformat()}
        st.download_button("Download assessment (JSON)", json.dumps(report, indent=2),
                           "p56_assessment.json", "application/json")

        st.markdown("### U4 — Visual and audio summary")
        category = answer.get("eu_category", "Unknown")
        fig = px.pie(pd.DataFrame({"Assessment":["EU AI Act","DPDP review"],"Status":[category,"Review obligations"]}),
                     names="Assessment", values=[1,1], title="P56 assessment overview")
        st.plotly_chart(fig, use_container_width=True)
        summary = (f"EU AI Act result: {category}. {answer.get('eu_reasoning','')} "
                   f"India DPDP review: {answer.get('dpdp_assessment','')} "
                   f"Recommended actions: {'; '.join(answer.get('recommended_actions',[]))}")
        if st.button("Generate audio explanation"):
            try:
                audio = io.BytesIO()
                gTTS(text=summary[:4500], lang="en").write_to_fp(audio)
                st.audio(audio.getvalue(), format="audio/mp3")
                st.download_button("Download audio MP3", audio.getvalue(), "p56_explanation.mp3", "audio/mpeg")
            except Exception as exc:
                st.error(f"Audio generation failed: {exc}")

with tab_cases:
    st.subheader("10 demonstration use cases")
    st.warning("These are indicative teaching examples, not teacher-approved legal conclusions. Your teacher must review and sign off the CSV.")
    rows = []
    for n, (title, desc, eu, dpdp) in enumerate(CASES, 1):
        rows.append({"No":n,"Use case":title,"Description":desc,"Indicative EU classification":eu,
                     "DPDP screening":dpdp,"Teacher verified":"PENDING","Teacher comments":"","Teacher name":"","Date":""})
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button("Download teacher verification CSV", frame.to_csv(index=False),
                       "p56_teacher_verification.csv", "text/csv")
    st.caption("Do not label these cases teacher-verified until your instructor has reviewed and approved them.")

with tab_retrieve:
    st.subheader("U3 — Sentence-BERT semantic regulation search")
    query = st.text_input("Describe a legal topic or AI use case", placeholder="Example: AI ranks job applicants")
    if query:
        for item in retrieve(query, documents, model, index, 8):
            with st.expander(f"{item['jurisdiction']} | {item['reference']} | similarity {item['similarity']:.3f}"):
                st.markdown(f"**{item['title']}**")
                st.write(item["text"])
                st.caption(f"Evidence ID: {item['id']}")
                st.markdown(f"[Open source]({item['url']})")

with tab_about:
    st.subheader("U1–U5 project mapping")
    st.markdown("""
**U1 — Prompt:** `build_prompt()` requires reasons, uncertainty and evidence IDs.

**U2 — LLM:** Gemini receives the use case plus retrieved legal passages and returns structured JSON.

**U3 — Retrieval:** Sentence-BERT (`all-MiniLM-L6-v2`) creates embeddings; FAISS finds semantically similar passages.

**U4 — Infographic + TTS:** Plotly creates a visual summary and gTTS creates an audio explanation.

**U5 — Citations:** Each finding is expected to return evidence IDs. The app displays the matching source name, article/section and official URL next to that finding.
""")
    st.code("User input → U1 prompt → U3 Sentence-BERT/FAISS → U2 Gemini → U5 citations → U4 chart/audio", language="text")
    st.markdown(f"- [EU AI Act official text]({EU_URL})\n- [European Commission AI Act overview]({EU_GUIDE})\n- [India Code DPDP Act 2023]({DPDP_URL})\n- [MeitY DPDP Rules 2025 notification]({RULES_URL})")
    st.warning("This is an educational screening tool, not legal advice or a compliance certificate. The built-in passages are concise retrieval aids, not the complete statutes. Verify full legal text, exceptions, commencement dates, and facts with a qualified reviewer.")
