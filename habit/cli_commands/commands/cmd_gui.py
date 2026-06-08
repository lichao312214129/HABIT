# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Command implementation to launch the HABIT Web GUI.
Spawns Gradio in a subprocess and opens the local browser.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import click

def run_gui_server(host: str = "127.0.0.1", port: int = 8501) -> None:
    """
    Launches the Gradio Web GUI server as a subprocess and opens
    the default web browser automatically.

    Args:
        host (str): IP address to bind the local server.
        port (int): Port number for the web server to listen on.
    """
    # Locate the GUI entrance script inside the habit package
    gui_entry_path: Path = Path(__file__).parent.parent.parent / "gui" / "app.py"
    
    if not gui_entry_path.exists():
        click.secho(f"Error: GUI main entrance not found at {gui_entry_path}", fg="red", err=True)
        sys.exit(1)
        
    click.secho("===================================================", fg="cyan")
    click.secho("   HABIT - Habitat Analysis Toolkit Web GUI       ", fg="cyan", bold=True)
    click.secho("===================================================", fg="cyan")
    click.secho(f"Starting local server at http://{host}:{port}...", fg="green")
    
    # Automatically open local browser
    webbrowser.open(f"http://{host}:{port}")
    
    # Spawn Python directly since app.py calls demo.launch()
    # We can pass host and port via environment variables or just let it use defaults
    # For now, we will set environment variables that Gradio respects, or we can modify app.py to read them.
    # Gradio respects GRADIO_SERVER_NAME and GRADIO_SERVER_PORT
    env = os.environ.copy()
    env["GRADIO_SERVER_NAME"] = host
    env["GRADIO_SERVER_PORT"] = str(port)
    
    cmd: list[str] = [
        sys.executable,
        str(gui_entry_path)
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        click.secho("\nWeb GUI server stopped gracefully.", fg="yellow")
    except Exception as e:
        click.secho(f"\nFailed to launch Web GUI: {e}", fg="red", err=True)
