# P56 — AI Regulation Use-Case Classifier

A beginner-friendly Streamlit project that screens AI use cases under:
- EU AI Act risk tiers
- India's Digital Personal Data Protection (DPDP) Act 2023

## U1–U5
1. **U1 Prompt:** a structured prompt asks for reasons, uncertainty and evidence IDs.
2. **U2 LLM:** Gemini generates the assessment in JSON.
3. **U3 Retrieval:** Sentence-BERT + FAISS finds relevant regulation passages.
4. **U4 Visual/audio:** Plotly infographic and gTTS audio explanation.
5. **U5 Citations:** each finding displays its evidence ID, article/section and source URL.

## Run on your computer
Install Python 3.10 or 3.11, then in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Paste your Gemini API key in the app's sidebar. Do not share it or commit it to GitHub.

## Deploy on Streamlit Community Cloud
1. Push these files to the GitHub repository `Vibhutishelke/P56-AI-Regulation-Classifier` on branch `main`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **New app** (or open the existing app), choose repository `Vibhutishelke/P56-AI-Regulation-Classifier`, branch `main`, and file `app.py`.
4. Deploy/redeploy.
5. In app settings → **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
   ```
6. Save and wait for the app to restart. Test one case and the regulation search tab.

## Teacher verification
The 10 cases in the app and `p56_teacher_verification.csv` are indicative examples. They are **not** marked teacher-verified. Give the CSV to your instructor, ask them to check the sources and classifications, and have them fill the teacher name, date, comments and approval status.

## Important
- Revoke any API key that was previously exposed in a screenshot or public repository. Create a new key and keep it in Streamlit Secrets.
- Built-in legal passages are short retrieval aids, not a complete copy of the statutes. Upload current official consolidated PDFs/TXT files in the sidebar for deeper semantic retrieval.
- The DPDP Act/Rules have phased commencement. Check provision-specific dates.
- The app is for education only, not legal advice or a compliance certificate.
- Do not upload real personal or confidential data to a public demo.
