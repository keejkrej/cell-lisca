from argparse import ArgumentParser
from cell_filter.extract import Extractor
import logging


def main():
    p = ArgumentParser()
    p.add_argument("--patterns", default="data/20250806_patterns_after.nd2")
    p.add_argument("--cells", default="data/20250806_MDCK_timelapse_crop_fov0004.nd2")
    p.add_argument("--nuclei-channel", type=int, default=1)
    p.add_argument("--filter-results", default="data/analysis/")
    p.add_argument("--output", default="data/analysis/")
    p.add_argument("--min-frames", type=int, default=20)
    p.add_argument("--max-gap", type=int, default=6, help="Maximum frame gap before splitting sequences")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    extractor = Extractor(
        patterns_path=args.patterns,
        cells_path=args.cells,
        output_folder=args.output,
        nuclei_channel=args.nuclei_channel,
    )
    extractor.extract(filter_results_dir=args.filter_results, min_frames=args.min_frames, max_gap=args.max_gap)


if __name__ == "__main__":
    main()
