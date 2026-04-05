# FYP News Collection

## Run After Cloning

1. Clone and enter the project:

```bash
git clone <your-repo-url>
cd FYP
```

2. Create and activate a virtual environment:

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Register a Jupyter kernel for this environment:

```bash
python -m ipykernel install --user --name fyp-news --display-name "Python (fyp-news)"
```

5. Start Jupyter:

```bash
jupyter notebook
```

6. Open `src/data_collection.ipynb`, select kernel `Python (fyp-news)`, and run all cells.

## Notes

- Run Jupyter from the project root so imports like `from src.news_scraper import ...` work.
- Keep output paths relative to the repo (for example `src/malaysia_news_since_2023.csv`).

## License

This repository's original code is released under the MIT License. See [LICENSE](LICENSE).

Third-party datasets, papers, model checkpoints, and other externally sourced materials remain subject to their own original licenses and terms, and are not relicensed by this repository.
