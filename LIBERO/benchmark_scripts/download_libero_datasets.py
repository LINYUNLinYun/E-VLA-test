import init_path
import argparse
import os
import time

from huggingface_hub import snapshot_download

import libero.libero.utils.download_utils as download_utils
from libero.libero import get_libero_path


HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=get_libero_path("datasets"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=[
            "all",
            "libero_goal",
            "libero_spatial",
            "libero_object",
            "libero_10",
            "libero_90",
            "libero_100",
        ],
        default="all",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face instead of original download links",
    )
    return parser.parse_args()


def download_from_huggingface(datasets, download_dir):
    if datasets == "libero_100":
        patterns = [
            "libero_10/*",
            "libero_90/*",
        ]
    elif datasets == "all":
        patterns = [
            "libero_goal/*",
            "libero_spatial/*",
            "libero_object/*",
            "libero_10/*",
            "libero_90/*",
        ]
    else:
        patterns = [f"{datasets}/*"]

    print(f"Hugging Face patterns: {patterns}")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=patterns,
    )


def main():

    args = parse_args()

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    if args.use_huggingface:
        print("Using Hugging Face as the download source")
        download_from_huggingface(
            datasets=args.datasets,
            download_dir=args.download_dir,
        )
    else:
        if args.datasets in ["libero_10", "libero_90"]:
            raise ValueError(
                "libero_10/libero_90 direct download is only supported with "
                "--use-huggingface. For original links, use --datasets libero_100."
            )

        print("Using original download links (note: these may expire soon)")
        input_str = input(
            "Download from original links may lead to failures. "
            "Do you want to continue? (y/n): "
        )
        if input_str.lower() != "y":
            print("Switching to Hugging Face as the download source...")
            download_from_huggingface(
                datasets=args.datasets,
                download_dir=args.download_dir,
            )
        else:
            download_utils.libero_dataset_download(
                download_dir=args.download_dir,
                datasets=args.datasets,
                use_huggingface=False,
            )

    # wait for 1 second
    time.sleep(1)
    print("\n\n\n")

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()