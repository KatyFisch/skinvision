import sys

from jupyterlab.labapp import LabApp


def main() -> None:
    LabApp.launch_instance(
        argv=[
            "--ServerApp.root_dir=.",
            "--LabApp.extension_manager=readonly",
            "notebooks/segmentation.ipynb",
            *sys.argv[1:],
        ]
    )
