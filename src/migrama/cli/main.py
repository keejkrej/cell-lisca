"""Unified CLI for all migrama modules."""

import typer

# Create main app
app = typer.Typer(help="Migrama: A comprehensive toolkit for micropatterned timelapse microscopy analysis")

# Add sub-apps for each module
pattern_app = typer.Typer(help="Detect and annotate micropatterns in microscopy images")
filter_app = typer.Typer(help="Analyze and extract micropatterned timelapse microscopy data")
extract_app = typer.Typer(help="Extract cropped sequences from filtered data")
graph_app = typer.Typer(help="Create region adjacency graphs and analyze T1 transitions")
tension_app = typer.Typer(help="Analyze tension maps from segmentation data")
viewer_app = typer.Typer(help="View NPY files with interactive GUI")

app.add_typer(pattern_app, name="pattern")
app.add_typer(filter_app, name="filter")
app.add_typer(extract_app, name="extract")
app.add_typer(graph_app, name="graph")
app.add_typer(tension_app, name="tension")
app.add_typer(viewer_app, name="viewer")

# Import commands from each module
from ..pattern.main import app as pattern_commands
from ..filter.main import app as filter_commands
from ..graph.main import app as graph_commands
from ..tension.cli import app as tension_commands

# Add commands to sub-apps
for cmd in pattern_commands.commands.values():
    pattern_app.command()(cmd)

for cmd in filter_commands.commands.values():
    filter_app.command()(cmd)

for cmd in graph_commands.commands.values():
    graph_app.command()(cmd)

for cmd in tension_commands.commands.values():
    tension_app.command()(cmd)

# Viewer command
@viewer_app.command()
def launch():
    """Launch the interactive NPY viewer."""
    from ..viewer.main import main
    main()

def main():
    """Main entry point for migrama CLI."""
    app()

if __name__ == "__main__":
    main()
