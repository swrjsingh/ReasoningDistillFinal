#!/usr/bin/env python
"""
Script to test if DeepSeek model loading works correctly.
Run this script before attempting the full data generation and filtering.
"""

import os
import sys
import logging
import argparse
import torch

# Add parent directory to path so we can import the custom model loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_model_loader import load_deepseek_model, test_deepseek_model_loading


def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepSeek model loading")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). If not specified, will use CUDA if available.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info(f"Testing DeepSeek model loading on {device}...")

    # Test model loading
    test_result = test_deepseek_model_loading(cache_dir=args.cache_dir, device=device)

    if test_result:
        logging.info("✓ SUCCESS: DeepSeek model loaded and tested successfully!")
        logging.info("You can now run the data generation and filtering scripts.")
        return 0
    else:
        logging.error("✗ FAILURE: DeepSeek model test failed.")
        logging.error("Please check the error messages above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
