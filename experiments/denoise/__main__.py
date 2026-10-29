"""Command line of the denoising experiment; see experiments/denoise/__init__.py."""
import argparse
from pathlib import Path

from . import __doc__ as DOC
from .data import CLASSES, prepare_store
from .noise import SOURCES


def main():
    parser = argparse.ArgumentParser(description=DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("store", "fit", "train", "evaluate", "table"))
    parser.add_argument("--source", choices=tuple(SOURCES), help="training-noise source (train, evaluate)")
    parser.add_argument("--classes", nargs="+", choices=CLASSES, metavar="CLASS",
                        help="PGNM classes to fit, e.g. S6_400 (default: all 15)")
    parser.add_argument("--presets", type=Path, help="PGNM preset folder (default: presets/denoise)")
    parser.add_argument("--store", type=Path, help="decoded training pairs (default: runs/denoise/store)")
    parser.add_argument("--checkpoint", type=Path, help="weights to evaluate (default: runs/denoise/<source>/model.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8, help="data workers; PGNM renders in them")
    test = parser.add_argument_group("quick test of the training loop")
    test.add_argument("--epochs", type=int, default=30)
    test.add_argument("--batch-size", type=int, default=128)
    test.add_argument("--batches-per-epoch", type=int)
    test.add_argument("--out", type=Path, help="run folder (default: runs/denoise/<source>)")
    args = parser.parse_args()
    if args.stage in ("train", "evaluate") and args.source is None:
        parser.error("--source is required for " + args.stage)
    if args.stage == "store":
        prepare_store(args.store)
    elif args.stage == "fit":
        from .noise import fit
        fit(args.classes)
    elif args.stage == "train":
        from .train import train
        train(args.source, workers=args.workers, device=args.device, presets=args.presets, store=args.store,
              out=args.out, epochs=args.epochs, batch_size=args.batch_size, batches_per_epoch=args.batches_per_epoch)
    elif args.stage == "evaluate":
        from .evaluate import evaluate
        evaluate(args.source, args.checkpoint, args.device)
    else:
        from .evaluate import print_table, table
        print_table(table())


if __name__ == "__main__":
    main()
