# SMO

Agentic second medical opinion system packaged as a Python project with a FastAPI interface, local saved-visit browsing, and optional paper evaluation scripts.

## Canonical Code

- `src/smo/`: application logic, retrieval, orchestration, formatting, and persistence.
- `src/smo/web/`: FastAPI app, templates, and static assets.
- `scripts/`: app launcher, API checker, and optional evaluation entrypoints.
- `data/guidelines/`: WHO/MSF guideline PDFs used to build the retrieval index.
- `data/samples/`: sample case inputs used by the project and paper.

## Recommended Environment

- Python `3.11`
- Windows PowerShell examples are shown below

## Setup

1. Create and activate a virtual environment.
2. Install the core app.
3. Copy `.env.example` to `.env`.
4. Fill in the API keys you want to use.
5. Place the guideline PDFs in `data/guidelines/`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install .
Copy-Item .env.example .env
```

Minimal `.env` values:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
SMO_EMBEDDING_MODEL=text-embedding-ada-002
```

Expected guideline filenames from the paper/notebook workflow:

- `guideline-170-en-61-113.pdf`
- `TUV_D1_RESPIRATORY GUIDELINES.pdf`

If needed, you can pin the files explicitly:

```env
SMO_GUIDELINE_PATHS='C:/Users/Ahmed/SMO/data/guidelines/guideline-170-en-61-113.pdf;C:/Users/Ahmed/SMO/data/guidelines/TUV_D1_RESPIRATORY GUIDELINES.pdf'
```

## Run The App

```powershell
.\scripts\run_app.ps1
```

Then open `http://127.0.0.1:8000`.

Practical note:

- The first live submission builds the FAISS index under `data/indices/` and will cost more than later runs.
- Later runs reuse that index unless you delete it or change retrieval settings.

## Working With Patient IDs

- Leave `Patient ID` blank to auto-generate an ID like `mso-xxxxxxxx`.
- Enter a patient ID manually if you want a stable case ID such as `P001`.
- Reuse the exact same patient ID for follow-up visits.

## Saved Visits And Paper Export

Every successful run is saved locally under `data/patient_records/`.

From the app you can:

- use `Browse Saved Visits` to list every saved visit for a patient ID
- open any saved visit without making new API calls
- use `Open Paper View` to open the cleaner export page for screenshots or print-to-PDF capture

Saved visits are local runtime artifacts and are ignored by Git by default.

## Optional Evaluation Scripts

The evaluation workflow is optional and uses live API calls.

Install the heavier evaluation stack only if you want to run the paper-style scripts:

```powershell
.\.venv\Scripts\python.exe -m pip install .[eval]
```

Run them with:

```powershell
.\.venv\Scripts\python.exe scripts\run_alignment_eval.py
.\.venv\Scripts\python.exe scripts\run_repeatability_eval.py
.\.venv\Scripts\python.exe scripts\run_cost_report.py
```

Evaluation artifacts are written to `outputs/evaluation/`.

Important note:

- A rerun should reproduce the same general evaluation trend, not necessarily identical scores or wording, because the scripts call live models.
- If you are already satisfied with the paper figures and do not want to spend tokens, you do not need to rerun the evaluation.

## Repository Layout

```text
.
|-- .env.example
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- guidelines/
|   |-- indices/
|   |-- patient_records/
|   `-- samples/
|-- scripts/
`-- src/
    `-- smo/
```
