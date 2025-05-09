import logging

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from opensynth.datasets.low_carbon_london import get_data as get_data_lcl
from src.opensynth.datasets.goiener import get_data as get_data_goiener

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

app = typer.Typer(context_settings=dict(max_content_width=800))


@app.command("get_lcl_data")
def get_lcl_data(
    download: Annotated[
        bool, typer.Option("--download", help="Downloads LCL data.")
    ] = False,
    split: Annotated[
        bool,
        typer.Option(
            "--split", help="Splits LCL households into training/ holdout set"
        ),
    ] = False,
    preprocess: Annotated[
        bool,
        typer.Option(
            "--preprocess",
            help="Preprocesses LCL data to create 48-half hour daily"
            "load profiles",
        ),
    ] = False,
):
    """
    Download, split and preprocess the Low Carbon London dataset.
    """
    get_data_lcl.get_lcl_data(
        download=download, split=split, preprocess=preprocess
    )

@app.command("get_goiener_data")
def get_goiener_data(
    download: Annotated[
        bool, typer.Option("--download", help="Downloads GoiEner data.")
    ] = False,
    split: Annotated[
        bool,
        typer.Option(
            "--split", help="Splits GoiEner households into training/ holdout set"
        ),
    ] = False,
    preprocess: Annotated[
        bool,
        typer.Option(
            "--preprocess",
            help="Preprocesses GoiEner data to create 48-half hour daily"
            "load profiles",
        ),
    ] = False,
):
    """
    Download, split and preprocess the Low Carbon London dataset.
    """
    get_data_goiener.get_goiener_data(
        download=download, split=split, preprocess=preprocess
    )


if __name__ == "__main__":
    app()
