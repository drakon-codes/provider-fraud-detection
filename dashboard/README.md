# Provider Intelligence dashboard

This folder includes two interfaces. The generated dashboard JSON is excluded from Git because it contains provider-level records. After running the main pipeline, run `python prepare_data.py` from this folder to create local snapshots before opening or deploying the static dashboard.

- `index.html`: static interactive dashboard for Vercel. It reads local JSON snapshots and uses browser JavaScript; no Python server or build step is needed. Run `python prepare_data.py` after completing the main pipeline to generate those snapshots before previewing or deploying.
- `app.py`: optional Streamlit version for local use.

## Deploy on Vercel

1. Import the repository into Vercel.
2. Set **Root Directory** to `dashboard`.
3. Set **Framework Preset** to **Other**.
4. Set **Build Command** to blank (skip build); set **Output Directory** to `.`.
5. Deploy.

If the underlying pipeline outputs change, refresh the local static snapshots with `python prepare_data.py`, then redeploy. Do not commit the generated JSON snapshots.

## Run locally

Open `index.html` through a local web server (browsers restrict `fetch()` from `file://`):

```powershell
python -m http.server 8000
```

Then visit `http://localhost:8000` while your current directory is `dashboard`.

For the Streamlit interface instead:

```powershell
python -m pip install streamlit plotly pandas
python -m streamlit run app.py
```
