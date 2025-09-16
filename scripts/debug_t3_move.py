"""
Debug helper: construct T3 and attempt to move its submodules and attributes to a target device
one at a time, printing progress and catching exceptions. Run this from the repo root with the
same Python environment you use to run the app.

Usage:
    python scripts/debug_t3_move.py --device cpu
    python scripts/debug_t3_move.py --device cuda:0

This will show the first failing attribute so you can paste the stack trace here.
"""
import sys
import traceback
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from chatterbox.models.t3.t3 import T3


def try_move(obj, device, name=None):
    try:
        # Prefer using torch.device
        dev = torch.device(device)
        if hasattr(obj, 'to'):
            obj.to(dev)
        elif torch.is_tensor(obj):
            obj.to(dev)
        else:
            # No obvious .to - skip
            pass
        return True, None
    except Exception as e:
        return False, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f"Constructing T3 model...")
    model = T3()
    print("Constructed T3, now probing submodules...")

    # 1) Try top-level to see original failure
    try:
        model.to(torch.device(args.device))
        print("Top-level .to succeeded on whole model")
    except Exception as e:
        print("Top-level .to FAILED")
        traceback.print_exc()

    # 2) Try moving children (named_children)
    for name, child in model.named_children():
        print(f"Moving child module: {name} ({type(child).__name__})...", end=' ')
        ok, err = try_move(child, args.device, name=name)
        if ok:
            print("OK")
        else:
            print("FAILED ->")
            traceback.print_exception(type(err), err, err.__traceback__)
            print("--- End failure for child ---")

    # 3) Inspect attributes on the main model for plain tensors / arrays
    print("\nInspecting model.__dict__ attributes for tensor-like objects...")
    for k, v in list(model.__dict__.items()):
        if isinstance(v, torch.Tensor) or hasattr(v, 'to'):
            print(f"Attempting to move attribute '{k}' of type {type(v).__name__}...", end=' ')
            ok, err = try_move(v, args.device, name=k)
            if ok:
                print("OK")
            else:
                print("FAILED ->")
                traceback.print_exception(type(err), err, err.__traceback__)
                print("--- End failure for attribute ---")

    print("Done. If you saw a failure above, please paste the output and traceback here.")


if __name__ == '__main__':
    main()
