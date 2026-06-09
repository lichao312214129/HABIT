# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
HABIT Web GUI main entry point (Gradio).
Launches the Gradio application and renders all workflow tabs.
"""

import os
import gradio as gr

from habit.gui.components.tab_dicom_sort import render_dicom_sort_tab
from habit.gui.components.tab_preprocess import render_preprocess_tab
from habit.gui.components.tab_habitat import render_habitat_tab
from habit.gui.components.tab_extract import render_extract_tab
from habit.gui.components.tab_ml import render_ml_tab
from habit.gui.components.tab_compare import render_compare_tab


def main(inbrowser: bool = False) -> None:
    """
    Main app runner: Gradio Blocks with six workflow tabs.

    Args:
        inbrowser: If True, open the default browser after Gradio binds the port.
    """
    custom_css = """
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    """

    with gr.Blocks(title="HABIT - Habitat Analysis Toolkit Web GUI", css=custom_css) as demo:
        gr.HTML("<div class='main-title'>HABIT Habitat Analysis Platform</div>")
        gr.HTML(
            "<div class='subtitle'>"
            "Habitat Analysis: Biomedical Imaging Toolkit — Web Graphical Interface"
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab("1. DICOM Sort"):
                render_dicom_sort_tab()
            with gr.Tab("2. Preprocessing"):
                render_preprocess_tab()
            with gr.Tab("3. Habitat Clustering"):
                render_habitat_tab()
            with gr.Tab("4. Feature Extraction"):
                render_extract_tab()
            with gr.Tab("5. Machine Learning"):
                render_ml_tab()
            with gr.Tab("6. Model Comparison"):
                render_compare_tab()

    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "8501"))
    open_browser: bool = inbrowser or os.environ.get("HABIT_GUI_INBROWSER", "").lower() in (
        "1",
        "true",
        "yes",
    )
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=open_browser,
    )


if __name__ == "__main__":
    main(inbrowser=True)
