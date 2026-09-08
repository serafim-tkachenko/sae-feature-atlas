"""Build the clean-runtime Colab orchestration notebook."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(
        dict(
            cell_type="code",
            metadata={},
            execution_count=None,
            outputs=[],
            source=text.splitlines(True),
        )
    )


md("""# Activation-conditioned SAE neighborhoods: real Gemma Scope experiment

Use **Runtime → Change runtime type → GPU**. A paid Colab L4/A100 runtime with at least 12 GB GPU RAM is recommended. This notebook uses an isolated, pinned Python 3.11 environment and CUDA 12.6 wheels; it does not depend on the notebook kernel's Python version. Start with the default pilot, or explicitly select the larger high-RAM experiment below.

Until the research branch is pushed, upload `sae_feature_atlas_source.zip` when the setup cell asks. The archive contains the local research code and dependency lock. After the branch is pushed, setup can clone it. No PR is required. Hugging Face access approval for Gemma must be obtained by the account holder; a Colab secret named `HF_TOKEN` is used when available. Drive mounting is optional and requires the normal Google consent prompt.

The notebook computes actual results. Synthetic test success never substitutes for model execution. Estimated resources scale with token count and observed L0; no parameter is silently reduced.
""")
md("## SETUP")
code("""import os
import sys
import subprocess
import zipfile
import pathlib
import shutil
import json
from IPython.display import display, Image, Markdown
from google.colab import files
REPO = "https://github.com/serafim-tkachenko/sae-feature-atlas.git"
REF = "research/activation-regimes"
ROOT = pathlib.Path("/content/sae-feature-atlas")
if not (ROOT / "src/sae_feature_atlas/scientific/run.py").exists():
    available = subprocess.run(["git", "ls-remote", "--heads", REPO, REF], capture_output=True, text=True)
    if available.returncode == 0 and available.stdout.strip():
        subprocess.run(["git", "clone", "--branch", REF, "--single-branch", REPO, str(ROOT)], check=True)
    else:
        print("Upload sae_feature_atlas_source.zip from the delivered analysis bundle.")
        uploaded = files.upload()
        archives = [name for name in uploaded if name.endswith(".zip")]
        if len(archives) != 1:
            raise ValueError("Upload exactly one source ZIP.")
        ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as z:
            for name in z.namelist():
                target = (ROOT / name).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError("Unsafe ZIP path")
            z.extractall(ROOT)
os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.9.28"], check=True)
subprocess.run(["uv", "python", "install", "3.11"], check=True)
subprocess.run(["uv", "venv", "--python", "3.11", "/content/sae-env"], check=True)
PY = "/content/sae-env/bin/python"
subprocess.run(["uv", "pip", "sync", "--python", PY, "--require-hashes", "requirements/colab.lock",
                "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy", "unsafe-best-match"], check=True)
subprocess.run(["uv", "pip", "install", "--python", PY, "--no-deps", "-e", "."], check=True)
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
except Exception:
    pass
print("Isolated research environment ready.")
""")
md("## CONFIG")
code("""# All experimental choices are exposed here. Leave defaults for the first run.
LARGE_RUN = False
USE_DRIVE = False
RUN_NAME = "gemma1b_regimes_colab_large" if LARGE_RUN else "gemma1b_regimes_colab_pilot"
MAX_TEXTS = 5000 if LARGE_RUN else 1000
MAX_SEQ_LEN = 512 if LARGE_RUN else 256
REGIMES = dict(seed=20260907, discovery_fraction=0.5,
               screen_features=2048 if LARGE_RUN else 512,
               max_candidates=48 if LARGE_RUN else 24,
               min_discovery_support=200, min_discovery_documents=30,
               posterior_threshold=0.9, min_regime_support=60,
               min_regime_documents=20, min_partner_support=10, neighbor_k=20,
               permutations=9999 if LARGE_RUN else 1999, bootstraps=1000 if LARGE_RUN else 300,
               position_bin=64, support_bin=16, match_log_caliper=0.35, gmm_n_init=5,
               delta_bic_threshold=10.0, min_component_weight=0.1, min_separation=2.0)
# Model google/gemma-3-1b-pt, native layer 13, 16k width, medium target L0=60,
# ALL positive activations, collection seed 42. No top-k cap.
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_ROOT = pathlib.Path("/content/drive/MyDrive/sae-regime-research/data")
    REPORTS_ROOT = pathlib.Path("/content/drive/MyDrive/sae-regime-research/reports")
else:
    DATA_ROOT, REPORTS_ROOT = ROOT / "data/processed", ROOT / "reports"
settings = dict(run_name=RUN_NAME, max_texts=MAX_TEXTS, max_seq_len=MAX_SEQ_LEN,
                regimes=REGIMES, data_root=str(DATA_ROOT), reports_root=str(REPORTS_ROOT))
pathlib.Path("colab_settings.json").write_text(json.dumps(settings, indent=2))
def stage(name):
    subprocess.run([PY, "scripts/colab_stage.py", name, "colab_settings.json"], check=True)
print(json.dumps(settings, indent=2))
""")
md("## SANITY CHECK")
code("""subprocess.run(["nvidia-smi"], check=True)
access_code = "from huggingface_hub import hf_hub_download; from huggingface_hub.errors import GatedRepoError; import sys\\ntry: hf_hub_download('google/gemma-3-1b-pt', 'config.json')\\nexcept GatedRepoError: sys.exit(42)"
access = subprocess.run([PY, "-c", access_code])
if access.returncode == 42:
    import getpass
    os.environ["HF_TOKEN"] = getpass.getpass("Approved Hugging Face Gemma token: ")
elif access.returncode:
    raise RuntimeError("Hugging Face access check failed; inspect the network error above.")
stage("sanity")
subprocess.run([PY, "-m", "pytest", "-q", "tests/test_scientific_regimes.py"], check=True)
""")
md(
    "## DATA COLLECTION\n\nThe first document validates the actual model/SAE dimensions, finite values, nonzero support and reconstruction error before proceeding. Collection checkpoints every 25 documents and checks hashes on resume. Nonfinite outputs fail explicitly. Model, SAE and corpus revisions are pinned in the run provenance."
)
code('stage("collect")\n')
md(
    "## BASE PIPELINE\n\nCompute stored/eligible token populations, positive-activation feature statistics and discovery GMM diagnostics. The focused scientific path omits global decoder PCA/UMAP and pruned graph generation because those are descriptive diagnostics, not evidence for the present hypothesis."
)
md(
    "## REGIME EXPERIMENT\n\nFit on discovery documents; evaluate frozen posterior assignments and complete raw partner membership on separate documents."
)
code('stage("analyze")\n')
md(
    "## NULL / CONTROL ANALYSIS\n\nThe preceding stage runs both conditional nulls and the strict delta-BIC control arm. The next stages run the separately labeled exploratory weak-separation arm and the one-observation-per-document sensitivity. These are preserved as separate analyses, not substituted for a failed strict control."
)
code('stage("controls")\nstage("robustness")\n')
md("## FIGURES")
code("""stage("report")
REPORT_DIR = REPORTS_ROOT / RUN_NAME
for path in sorted((REPORT_DIR / "figures").glob("*.png")):
    display(Image(filename=str(path)))
""")
md("## SCIENTIFIC SUMMARY")
code('display(Markdown((REPORT_DIR / "scientific_report.md").read_text()))\n')
md(
    "## EXPORT\n\nExport includes all final raw/derived data, provenance, source code, lock files, the Markdown/PDF paper and plots. Completed collection chunks remain in the run directory for recovery; the bundle includes the consolidated raw tables rather than duplicate chunks."
)
code("""subprocess.run([PY, "scripts/render_scientific_pdf.py", str(REPORT_DIR / "scientific_report.md")], check=True)
stage("export")
archive = ROOT / "outputs" / (RUN_NAME + "_analysis_bundle.zip")
if USE_DRIVE:
    shutil.copy2(archive, REPORTS_ROOT / archive.name)
    print("Bundle copied to Drive:", REPORTS_ROOT / archive.name)
else:
    files.download(str(archive))
""")

notebook = dict(
    nbformat=4,
    nbformat_minor=5,
    cells=cells,
    metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        accelerator="GPU",
        colab=dict(name="gemma_activation_regimes.ipynb"),
    ),
)
for i, cell in enumerate(cells):
    cell["id"] = f"regime-{i:02d}"
Path("notebooks/gemma_activation_regimes.ipynb").write_text(
    json.dumps(notebook, indent=1), encoding="utf-8"
)
