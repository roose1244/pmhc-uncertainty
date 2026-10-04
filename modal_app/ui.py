"""The PepShield Streamlit app, served publicly.

This is the same `app/streamlit_app.py` that runs locally. Nothing about the
model, the intervals or the figures changes here; only the hosting does.

The app resolves everything relative to its repo root, so the container puts
the code at /root and symlinks /root/data to the volume copy of the data.

    modal deploy modal_app/serve.py
"""

from __future__ import annotations

import os
import subprocess

import modal

from modal_app.common import VOL, app, image

ui_image = (
    image.pip_install("streamlit>=1.65", "requests>=2.31", "py3Dmol>=2.4", "certifi")
    .add_local_python_source("src", "modal_app")
    .add_local_dir("app", "/root/app")
    .add_local_dir("results", "/root/results")
)

CMD = (
    "streamlit run /root/app/streamlit_app.py"
    " --server.port 8000 --server.address 0.0.0.0 --server.headless true"
    " --server.enableCORS false --server.enableXsrfProtection false"
    " --browser.gatherUsageStats false"
)


@app.function(image=ui_image, volumes=VOL, timeout=3600, scaledown_window=1200)
@modal.concurrent(max_inputs=50)
@modal.web_server(8000, label="pepshield", startup_timeout=180)
def ui() -> None:
    os.environ["PMHC_MODELS"] = "/vol/models"
    if not os.path.exists("/root/data"):
        os.symlink("/vol/data", "/root/data")
    subprocess.Popen(CMD, shell=True)
