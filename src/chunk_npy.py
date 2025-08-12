import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Iterable, List, Tuple

import numpy as np


def list_npy_files_sorted(input_dir: str) -> List[str]:
    """Return a numerically sorted list of .npy file paths.

    Assumes files are named like "{number}.npy". Non-numeric stems are ignored.
    """
    files = []
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.endswith(".npy"):
            stem = os.path.splitext(entry.name)[0]
            try:
                num = int(stem)
            except ValueError:
                # Skip non-numeric filenames to avoid unintended ordering
                continue
            files.append((num, entry.path))

    files.sort(key=lambda x: x[0])
    return [path for _, path in files]


def infer_sample_shape_and_dtype(sample_path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    arr = np.load(sample_path, mmap_mode=None)
    shape = tuple(arr.shape)
    dtype = arr.dtype
    if len(shape) != 3:
        raise ValueError(
            f"Expected each npy file to have shape (2, 256, 256), got {shape} from {sample_path}"
        )
    return shape, dtype


def save_atomic_npy(output_path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dir_name = os.path.dirname(output_path) or "."
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".npy") as tmp:
        tmp_path = tmp.name
        # Close immediately so numpy can open and write
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, output_path)
    except Exception:
        # Best-effort cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _load_one(
    path: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    use_mmap: bool,
):
    arr = np.load(path, mmap_mode="r" if use_mmap else None)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(
            f"Shape mismatch at {path}: expected {expected_shape}, got {tuple(arr.shape)}"
        )
    if arr.dtype != expected_dtype:
        arr = arr.astype(expected_dtype, copy=False)
    return arr


def _load_many(
    paths: List[str],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    workers: int,
    io_mode: str,
    use_mmap: bool,
) -> Iterable[np.ndarray]:
    if workers <= 1:
        return (_load_one(p, expected_shape, expected_dtype, use_mmap) for p in paths)

    if io_mode == "thread":
        executor_cls = ThreadPoolExecutor
    elif io_mode == "process":
        executor_cls = ProcessPoolExecutor
    else:
        raise ValueError("io_mode must be 'thread' or 'process'")

    # Use executor.map to preserve order of inputs
    executor = executor_cls(max_workers=workers)
    return executor.map(
        _load_one,
        paths,
        (expected_shape,) * len(paths),
        (expected_dtype,) * len(paths),
        (use_mmap,) * len(paths),
    )


def chunk_and_stack(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 1000,
    start_index: int = 0,
    zero_pad: int = 0,
    workers: int = 1,
    io_mode: str = "thread",
    use_mmap: bool = False,
) -> None:
    all_files = list_npy_files_sorted(input_dir)
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    sample_shape, sample_dtype = infer_sample_shape_and_dtype(all_files[0])
    # Preallocate one chunk buffer to reuse for all full chunks
    chunk_buffer = np.empty((chunk_size,) + sample_shape, dtype=sample_dtype)

    total = len(all_files)
    num_full_chunks = total // chunk_size
    remainder = total % chunk_size

    def make_out_name(idx: int) -> str:
        name = f"{idx:0{zero_pad}d}.npy" if zero_pad > 0 else f"{idx}.npy"
        return os.path.join(output_dir, name)

    file_idx = 0
    out_idx = start_index

    # Process full chunks
    for _ in range(num_full_chunks):
        paths = all_files[file_idx : file_idx + chunk_size]
        # Load concurrently (or sequentially if workers=1)
        for i, arr in enumerate(
            _load_many(paths, sample_shape, sample_dtype, workers, io_mode, use_mmap)
        ):
            chunk_buffer[i] = arr
        file_idx += chunk_size

        out_path = make_out_name(out_idx)
        save_atomic_npy(out_path, chunk_buffer)
        print(f"Saved chunk {out_idx} to {out_path} ({file_idx}/{total} frames processed)")
        out_idx += 1

    # Process remainder, if any
    if remainder:
        rem_buffer = np.empty((remainder,) + sample_shape, dtype=sample_dtype)
        paths = all_files[file_idx : file_idx + remainder]
        for i, arr in enumerate(
            _load_many(paths, sample_shape, sample_dtype, workers, io_mode, use_mmap)
        ):
            rem_buffer[i] = arr
        file_idx += remainder
        out_path = make_out_name(out_idx)
        save_atomic_npy(out_path, rem_buffer)
        print(f"Saved chunk {out_idx} to {out_path} ({file_idx}/{total} frames processed)")

    print(
        f"Chunking complete. Wrote {num_full_chunks + (1 if remainder else 0)} files to {output_dir}."
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk many .npy files (each shaped like (2, 256, 256)) into stacked .npy files "
            "of shape (N, 2, 256, 256). Files are sorted numerically by stem before chunking."
        )
    )
    parser.add_argument("--input", required=True, help="Directory containing source .npy files")
    parser.add_argument("--output", required=True, help="Directory to write chunked .npy files")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of frames per chunk (default: 1000)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for output filenames (default: 0)",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=0,
        help="Zero pad width for output filenames (default: 0, i.e., no padding)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for loading inputs (suggest 4-16; default: 1)",
    )
    parser.add_argument(
        "--io-mode",
        choices=["thread", "process"],
        default="thread",
        help="Parallelism backend for loading (thread recommended)",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Use memory-mapped reads (reduces peak RAM during load)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    chunk_and_stack(
        input_dir=os.path.abspath(args.input),
        output_dir=os.path.abspath(args.output),
        chunk_size=args.chunk_size,
        start_index=args.start_index,
        zero_pad=args.zero_pad,
        workers=max(1, int(args.workers)),
        io_mode=args.io_mode,
        use_mmap=bool(args.mmap),
    )


if __name__ == "__main__":
    main(sys.argv[1:])


