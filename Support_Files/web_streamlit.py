"""
Entry file of the deployed Streamlit app (Streamlit Cloud runs: streamlit run Support_Files/web_streamlit.py).
It now opens the main app, app.py, in the project root.
The previous version of this page is kept in web_streamlit_old.py:
    streamlit run Support_Files/web_streamlit_old.py
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # project root, where app.py and app_backend.py are
sys.path.insert(0, str(ROOT))                       # so that app.py can import app_backend
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
