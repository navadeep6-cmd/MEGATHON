"""Interactive demo (Gradio): upload an image, review explained suggestions,
accept / reject them (feedback loop), and add new categories.

    python app/demo.py            # http://127.0.0.1:7860
"""

import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402  (installed with gradio)

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.feedback import apply_feedback, claim_serving_writer, log_feedback  # noqa: E402
from catsuggest.suggester import Suggester  # noqa: E402

CFG = load_config()
PATHS = Paths(CFG).ensure()
claim_serving_writer(PATHS.serve_lock)  # this process is the single writer of the serving index
SUG = Suggester(CFG, memory="serve")


COLUMNS = ["category", "score", "type", "why"]


def _num(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def run(image, title, description, date, lat, lon):
    if image is None:
        return gr.update(), gr.update(), gr.update(choices=[], value=[]), {}, "Upload an image first."
    res = SUG.suggest(image.convert("RGB"), title=title or "", description=description or "", date=date or "",
                      lat=_num(lat), lon=_num(lon), cache_query=True)
    rows = pd.DataFrame([[s["category"], s["score"], s["type"], s["why"]] for s in res["suggestions"]],
                        columns=COLUMNS)
    gallery = []
    for n in res["neighbours"]:
        src = n["path"] if n["path"] and os.path.exists(n["path"]) else n["thumb_url"]
        if src:
            gallery.append((src, "%.2f  %s" % (n["similarity"], "; ".join(n["categories"][:3]))))
    names = [s["category"] for s in res["suggestions"]]
    state = {"query_id": res["query_id"], "suggestions": res["suggestions"]}
    info = "query %s - %d suggestions in %.0f ms. Tick the correct ones and press Submit." % (
        res["query_id"], len(names), res["timing_ms"]["total"])
    return rows, gallery, gr.update(choices=names, value=[]), state, info


def submit(selected, state):
    if not state or not state.get("suggestions"):
        return "Nothing to submit."
    chosen = set(selected or [])
    for s in state["suggestions"]:
        log_feedback(PATHS.feedback_log, state["query_id"], s["category"], s["category"] in chosen,
                     s["score"], s["type"], source="demo")
    stats = apply_feedback(SUG, PATHS)  # learn immediately: accepted image joins the k-NN index
    return ("Logged %d accepted / %d rejected. Index updated now (images added %d, rows updated %d) - "
            "upload a similar image to see the effect. Rejects are kept as hard negatives for the next retrain."
            % (len(chosen), len(state["suggestions"]) - len(chosen), stats["images_added"], stats["rows_updated"]))


def add_category(name):
    if not name or not name.strip():
        return "Enter a category name."
    cid = SUG.add_category(name)
    SUG.save()
    return "Category '%s' is available now (id %d) - no retraining needed." % (SUG.store.names[cid], cid)


with gr.Blocks(title="Commons category suggester", analytics_enabled=False) as demo:
    gr.Markdown("## Wikimedia Commons - category suggestions\n"
                "Suggestions are for **human review**; nothing is edited automatically. "
                "Similar already-categorised images are shown as the explanation.")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Image")
            title = gr.Textbox(label="File name / title (optional)")
            description = gr.Textbox(label="Description (optional)", lines=2)
            date = gr.Textbox(label="Date taken (optional, e.g. 2019-05-04)")
            with gr.Row():
                lat = gr.Textbox(label="Latitude (optional)")
                lon = gr.Textbox(label="Longitude (optional)")
            go = gr.Button("Suggest categories", variant="primary")
        with gr.Column(scale=2):
            table = gr.Dataframe(headers=COLUMNS, label="Suggestions",
                                 interactive=False, wrap=True)
            gallery = gr.Gallery(label="Most similar categorised images (explanation)", columns=4, height=320)
            picks = gr.CheckboxGroup(label="Correct categories", choices=[])
            submit_btn = gr.Button("Submit feedback")
            status = gr.Markdown()
    with gr.Row():
        new_cat = gr.Textbox(label="Add a new category (usable immediately)")
        add_btn = gr.Button("Add category")
    state = gr.State({})
    go.click(run, [image, title, description, date, lat, lon], [table, gallery, picks, state, status])
    submit_btn.click(submit, [picks, state], [status])
    add_btn.click(add_category, [new_cat], [status])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[PATHS.images])
