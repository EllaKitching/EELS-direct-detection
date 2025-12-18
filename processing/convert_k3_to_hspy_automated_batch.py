import os
import h5py
import numpy as np
import hyperspy.api as hs
from tqdm import tqdm
import gc
import argparse

# Convert K3 DM5 files to HyperSpy format using raw binary data.
# Expects simultaneously acquired ADF/EDS/EELS (any combination) with same metadata params, otherwise metadata will be incorrect.
# Run script in command line from base directory containing InSitu folders.

# Written by Ella Kitching, Oct 2025, based on raw/dm5 file conversion code by Mohsen Danaie.

def detect_dataset_type(filename):
    """Return (signaltype, label) based on filename."""
    name = filename.lower()
    if "si_eels ll" in name:
        return "EELS", "LL"
    elif "si_eels hl" in name:
        return "EELS", "HL"
    elif "si_eds" in name:
        return "EDS_TEM", "EDS"
    elif "adf" in name:
        return "Signal2D", "ADF"
    else:
        return None, None


def process_datasets(base_path, args):
    """Process all InSitu folders and convert DM5/raw files to HyperSpy format."""
    print(f"Base path: {base_path}")
    print(f"Pixel size: {args.pixel_size} nm")
    print(f"EELS spectrum scale: {args.eels_spectrum_scale} eV/channel")
    print(f"EELS HL offset: {args.eels_hl_offset} eV")
    print(f"EELS LL offset: {args.eels_ll_offset} eV")
    print(f"EDS spectrum scale: {args.eds_spectrum_scale} keV/channel\n")

    #  find all InSitu folders and datasets recursively 
    insitu_folders = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.startswith("InSitu"):
                insitu_folders.append(os.path.join(root, d))

    insitu_folders = sorted(insitu_folders)
    if not insitu_folders:
        print("No InSitu folders found.")
        return
    
    print(f"Found {len(insitu_folders)} InSitu folder(s)\n")

    # set up base output directory
    save_dir = os.path.join(base_path, "hspy")
    os.makedirs(save_dir, exist_ok=True)

    # start processing each InSitu folder, skip if issues with metadata/files and print for debugging
    for insitu_folder in insitu_folders:

        rel_path = os.path.relpath(insitu_folder, base_path)
        dm5_files = [f for f in os.listdir(insitu_folder) if f.lower().endswith(".dm5") and f.startswith("STEM SI_")]

        if not dm5_files:
            print(f"Warning: No STEM SI_*.dm5 files in {rel_path}, skipping...")
            continue

        for dm5_file in dm5_files:
            dm5_path = os.path.join(insitu_folder, dm5_file)
            raw_path = os.path.splitext(dm5_path)[0] + ".raw"

            if not os.path.exists(raw_path):
                print(f"Warning: Missing RAW for {dm5_file}, skipping...")
                continue

            signaltype, label = detect_dataset_type(dm5_file)
            if signaltype is None:
                print(f"Warning: Unknown type for {dm5_file}, skipping...")
                continue

            folder_name = rel_path.replace(os.sep, "_").replace(" ", "_")
            stack_path = os.path.join(save_dir, f"{folder_name}_{label}_stack.hspy")
            summed_path = os.path.join(save_dir, f"{folder_name}_{label}_summed.hspy")

            # skip if already fully converted to enable restarting if issues encountered partway
            if os.path.exists(stack_path) and os.path.exists(summed_path):
                print(f"Skipping {dm5_file} as already converted.")
                continue
            elif os.path.exists(stack_path) or os.path.exists(summed_path):
                print(f"Partial conversion found for {dm5_file}, reprocessing...")

            print(f"\nProcessing {rel_path} as {dm5_file} ({signaltype}, {label})")

            # start conversion and reading data from dm5 to ensure correct shape and num of frames
            with h5py.File(dm5_path, "r") as f:
                data = f["ImageList/[1]/ImageData/Data"][()]
            print(f"Loaded DM5 shape: {data.shape}, dtype: {data.dtype}")

            array = np.fromfile(raw_path, dtype=data.dtype, count=-1, offset=0)
            frame_size = np.prod(data.shape)
            n_frames = int(array.size // frame_size)
            valid_size = int(n_frames) * int(frame_size)
            # trim excess data if present, as can happen if last frame incomplete
            array_trimmed = array[:valid_size]
            array_reshaped = array_trimmed.reshape((n_frames, *data.shape))
            print(f"Detected {n_frames} frames\n")

            # set up hyperspy signals and axes
            signals = []
            if "ADF" in dm5_file:
                s = hs.signals.Signal2D(array_reshaped)
                print(f"Converting {label} as Signal2D, shape {s.data.shape}")
                s.axes_manager.signal_axes[0].name = "y"
                s.axes_manager.signal_axes[0].units = "nm"
                s.axes_manager.signal_axes[0].scale = args.pixel_size
                s.axes_manager.signal_axes[1].name = "x"
                s.axes_manager.signal_axes[1].units = "nm"
                s.axes_manager.signal_axes[1].scale = args.pixel_size
                signals = [s]

            else:
                for i in tqdm(range(n_frames), desc=f"Converting {label} frames", unit="frame"):
                    # order axes correctly for Signal1D with image based on expected hyperspy order
                    frame_data = np.transpose(array_reshaped[i], (1, 2, 0))
                    s = hs.signals.Signal1D(frame_data)
                    s.set_signal_type(signaltype)

                    # set up spectral axes depending on signal type
                    if signaltype == "EELS":
                        s.axes_manager.signal_axes[0].name = "Energy Loss"
                        s.axes_manager.signal_axes[0].units = "eV"
                        s.axes_manager.signal_axes[0].scale = args.eels_spectrum_scale
                        offset = args.eels_hl_offset if "HL" in dm5_file else args.eels_ll_offset
                        s.axes_manager.signal_axes[0].offset = offset
                    elif signaltype == "EDS_TEM":
                        s.axes_manager.signal_axes[0].name = "Energy"
                        s.axes_manager.signal_axes[0].units = "keV"
                        s.axes_manager.signal_axes[0].scale = args.eds_spectrum_scale
                        s.axes_manager.signal_axes[0].offset = 0.0
                    
                    # set up navigation axes (will be same as data is simultanously acquired)
                    s.axes_manager.navigation_axes[0].name = "y"
                    s.axes_manager.navigation_axes[0].units = "nm"
                    s.axes_manager.navigation_axes[0].scale = args.pixel_size
                    s.axes_manager.navigation_axes[1].name = "x"
                    s.axes_manager.navigation_axes[1].units = "nm"
                    s.axes_manager.navigation_axes[1].scale = args.pixel_size

                    signals.append(s)

            # set up saving for summed and stack files, so each frame can be split during postprocessing if need be
            stack = hs.stack(signals)
            summed = signals[0].deepcopy()
            for signal in signals[1:]:
                summed.data += signal.data

            print(f"Saving to {save_dir}")
            summed.save(summed_path, overwrite=True)
            stack.save(stack_path, overwrite=True)

            # clean up to free memory
            del data, array, array_trimmed, array_reshaped, signals, stack, summed
            if 's' in locals():
                del s
            gc.collect()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert K3 DM5 files to HyperSpy format from accompanying raw data"
    )
    parser.add_argument(
        "-p", "--path", default=os.getcwd(),
        help="Base path with InSitu folders (default: current directory)"
    )
    parser.add_argument(
        "--pixel-size", type=float, default=0.15,
        help="ADF/SI image pixel size in nm (default: 0.15)"
    )
    parser.add_argument(
        "--eels-spectrum-scale", type=float, default=0.18,
        help="EELS spectrum scale in eV/channel (default: 0.18)"
    )
    parser.add_argument(
        "--eels-hl-offset", type=float, default=314.0,
        help="EELS high loss offset in eV (default: 314.0)"
    )
    parser.add_argument(
        "--eels-ll-offset", type=float, default=-36.0,
        help="EELS low loss offset in eV (default: -36.0)"
    )
    parser.add_argument(
        "--eds-spectrum-scale", type=float, default=0.01,
        help="EDS spectrum scale in keV/channel (default: 0.01)"
    )
    
    args = parser.parse_args()
    process_datasets(args.path, args)


if __name__ == "__main__":
    main()
