# streamlit_app.py
import streamlit as st
import subprocess, tempfile, datetime
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

st.set_page_config(
    page_title="Mathematical Pattern Discovery Engine",
    layout="wide",
)

st.title("📐 Mathematical Pattern Discovery Engine")
st.markdown("_An IIT-Jodhpur research project_")

uploaded = st.file_uploader("Upload CSV / TXT", type=["csv", "txt"])

def run_agent(path):
    """Call your CLI and return raw stdout."""
    proc = subprocess.run(
        ["python", "analyze.py", "--data", path],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout

if uploaded:
    with st.spinner("Running RL agent…"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read()); tmp.close()

        try:
            raw = run_agent(tmp.name)
        except Exception as e:
            st.error(e)
            st.stop()

    st.subheader("Raw output")
    st.code(raw, language="text")

    # --- minimal parsing for demo (reuse parser you already wrote) ----
    import re, json
    reward = json.loads(
        re.search(r"Reward break-down:\s*({.*})", raw).group(1).replace("'", '"')
    )
    st.subheader("Reward breakdown")
    st.json(reward)

    # chart of top relations
    rel_lines = re.findall(r"(.*?)→(.*?)\s*deg=\d+\s*R²=([\d.]+)", raw)
    if rel_lines:
        plt.figure(figsize=(5,2.5))
        labels = [f"{src.strip()}→{dst.strip()}" for src,dst,_ in rel_lines]
        vals   = [float(r2) for *_, r2 in rel_lines]
        plt.barh(labels, vals); plt.gca().invert_yaxis()
        st.pyplot(plt)

    # --- PDF report download -----------------------------------------
    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(pdf_tmp.name, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 11)
    for line in raw.splitlines():
        text.textLine(line)
    c.drawText(text); c.showPage(); c.save()

    with open(pdf_tmp.name, "rb") as f:
        st.download_button("Download PDF", f, file_name="report.pdf")
