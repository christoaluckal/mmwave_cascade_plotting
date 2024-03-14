"""Format MMWCAS-RF-EVM & MMWCAS-DSP-EVM kit Recordings.

    @author: AMOUSSOU Z. Kenneth
    @date: 13-08-2022
"""
from typing import Optional
from datetime import timedelta, datetime
import os
import glob
import argparse
import sys
import matplotlib.pyplot as plt
# 3d plotting
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
import pickle

__VERSION__: str = "0.1"
__COPYRIGHT__: str = "Copyright (C) 2022, RWU-RADAR Project"


def getInfo(idx_file: str):
    """Get information about the recordings.

    The "*_idx.bin" files along the sample files gather usefule
    information aout the dataset.

    The structure of the "*_idx.bin" file is as follow:

    ---------------------------------------------------------------------------
         File header in *_idx.bin:
             struct Info
             {
                 uint32_t tag;
                 uint32_t version;
                 uint32_t flags;
                 uint32_t numIdx;       // number of frames
                 uint64_t dataFileSize; // total data size written into file
             };

         Index for every frame from each radar:
             struct BuffIdx
             {
                 uint16_t tag;
                 uint16_t version; /*same as Info.version*/
                 uint32_t flags;
                 uint16_t width;
                 uint16_t height;

                 /*
                  * For image data, this is pitch. For raw data, this is
                  * size in bytes per metadata plane
                  */
                 uint32_t pitchOrMetaSize[4];

                /*
                 * Total size in bytes of the data in the buffer
                 * (sum of all planes)
                */
                 uint32_t size;
                 uint64_t timestamp; // timestamp in ns
                 uint64_t offset;
             };

        Source: Example matlab script provided by Texas Instrument
    ---------------------------------------------------------------------------

    Arguemnt:
        idx_file: Path to an index file from any of the cascaded chip

    Return:
        Tuple containing respectively the number of valid frames recorded
        and the size of the data file
    """
    # Data type based on the structure of the file header
    dt = np.dtype([
        ("tag", np.uint32),
        ("version", np.uint32),
        ("flags", np.uint32),
        ("numIdx", np.uint32),
        ("size", np.uint64),
    ])
    header = np.fromfile(idx_file, dtype=dt, count=1)[0]

    dt = np.dtype([
        ("tag", np.uint16),
        ("version", np.uint16),
        ("flags", np.uint32),
        ("width", np.uint16),
        ("height", np.uint16),

        ("_meta0", np.uint32),
        ("_meta1", np.uint32),
        ("_meta2", np.uint32),
        ("_meta3", np.uint32),

        ("size", np.uint32),
        ("timestamp", np.uint64),
        ("offset", np.uint64),
    ])

    data = np.fromfile(idx_file, dtype=dt, count=-1, offset=24)
    timestamps = np.array([
        (datetime.now() + timedelta(seconds=log[-2] * 1e-9)).timestamp()
        for log in data
    ])

    return header[3], header[4], timestamps


def load(inputdir: str, device: str):
    """Load the recordings of the radar chip provided in argument.

    Arguments:
        inputdir: Input directory to read the recordings from
        device: Name of the device

    Return:
        Dictionary containing the data and index files
    """
    # Collection of the recordings data file
    # They all have the "*.bin" ending
    recordings: dict[str, list[str]] = {
        "data": glob.glob(f"{inputdir}{os.sep}{device}*data.bin"),
        "idx": glob.glob(f"{inputdir}{os.sep}{device}*idx.bin")
    }
    recordings["data"].sort()
    recordings["idx"].sort()

    if (len(recordings["data"]) == 0) or (recordings["idx"] == 0):
        print(f"[ERROR]: No file found for device '{device}'")
        return None
    elif len(recordings["data"]) != len(recordings["idx"]):
        print(
            f"[ERROR]: Missing {device} data or index file!\n"
            "Please check your recordings!"
            "\nYou must have the same number of "
            "'*data.bin' and '*.idx.bin' files."
        )
        return None
    return recordings


def toframe(
        mf: str, sf0: str, sf1: str, sf2: str,
        ns: int, nc: int, nf: int,
        output: str = ".",
        start_idx: int = 0) -> int:
    """Re-Format the raw radar ADC recording.

    The raw recording from each device is merge together to create
    separate recording frames corresponding to the MIMO configuration.

    Arguments:
        mf: Path to the recording file of the master device
        sf0: Path to the recording file of the first slave device
        sf1: Path to the recording file of the second slave device
        sf2: Path to the recording file of the third slave device

        ns: Number of ADC samples per chirp
        nc: Number of chrips per frame
        nf: Number of frames to generate

        output: Path to the output folder where the frame files would
                be written

        start_idx: Index to start numbering the generated files from.

    Return:
        The index number of the last frame generated

    Note:
        Considering the AWR mmwave radar kit from Texas Instrument used,
        The following information can be infered:
            - Number of cascaded radar chips: 4
            - Number of RX antenna per chip: 4
            - Number of TX antenna per chip: 3
            - Number of signal measured per ADC samples: 2
                - In-phase signal (I)
                - Quadrature signal (Q)
    """
    # Number of waveform measured
    # 2-bytes (signed integer) for I (In-phase signal)
    # 2.bytes (signed integer) for Q (Quadrature signal)
    nwave: int = 2

    # Number of cascaded radar chips
    nchip: int = 4

    # Number of TX antenna
    ntx: int = 3

    # Number of RX antenna per chip
    nrx: int = 4

    # Number of frame to skip at the beginning of the recording
    nf_skip: int = 0

    # Index used to number frames
    fk: int = start_idx

    master_frames = []

    for fidx in range(nf_skip, nf):
        # Number of items to read (here items are 16-bit integer values)
        nitems: int = nwave * ns * nc * nrx * ntx * nchip
        # Offet to read the bytes of a given frame
        # The multiplication by "2" is to count for the size of 16-bit integers
        offset: int = fidx * nitems * 2

        dev1 = np.fromfile(mf, dtype=np.int16, count=nitems, offset=offset)
        dev2 = np.fromfile(sf0, dtype=np.int16, count=nitems, offset=offset)
        dev3 = np.fromfile(sf1, dtype=np.int16, count=nitems, offset=offset)
        dev4 = np.fromfile(sf2, dtype=np.int16, count=nitems, offset=offset)

        dev1 = dev1.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev2 = dev2.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev3 = dev3.reshape(nc, ntx * nchip, ns, nrx, 2)
        dev4 = dev4.reshape(nc, ntx * nchip, ns, nrx, 2)

        dev1 = np.transpose(dev1, (1, 3, 0, 2, 4))
        dev2 = np.transpose(dev2, (1, 3, 0, 2, 4))
        dev3 = np.transpose(dev3, (1, 3, 0, 2, 4))
        dev4 = np.transpose(dev4, (1, 3, 0, 2, 4))

        master_frames.append(dev1)

    return master_frames

def reordering(frames):

    data_dict = {
        'master_frames': None,
        'txrx_frame_collation': None,
        'txrx_frame_collation_diff': None,
        'frame_collate_diff': None
    }

    master_frames = np.array(frames)
    master_frames = master_frames[...,0]+1j*master_frames[...,1]


    print("Original Frame Data shape: ", master_frames.shape)

    # Inserting 0s to account for per chirp downtime
    zeroes = np.zeros((master_frames.shape[0], master_frames.shape[1], master_frames.shape[2], master_frames.shape[3], 152))
    ones = np.ones((master_frames.shape[0], master_frames.shape[1], master_frames.shape[2], master_frames.shape[3], 152))
    complex_ = ones + 1j*zeroes

    print("Chirp zeroes (1+0j) shape: ", complex_.shape)

    master_frames = np.concatenate((master_frames, complex_), axis=4)

    data_dict['master_frames'] = master_frames

    print("Frame shape with chirp zeroes: ", master_frames.shape)

    collate_chiprs = np.copy(master_frames).reshape(-1, master_frames.shape[1], master_frames.shape[2], master_frames.shape[3]*master_frames.shape[4])

    # txrx_frame_collation = np.fft.fftn(np.transpose(collate_chiprs, (1,2,0,3)))
    txrx_frame_collation = np.transpose(collate_chiprs, (1,2,0,3))

    print("TxRx Collation shape: ", txrx_frame_collation.shape)

    zeroes = np.zeros((txrx_frame_collation.shape[0], txrx_frame_collation.shape[1], txrx_frame_collation.shape[2], 25446))
    ones = np.ones((txrx_frame_collation.shape[0], txrx_frame_collation.shape[1], txrx_frame_collation.shape[2], 25446))
    complex_ = ones + 1j*zeroes

    print("Frame zeroes (1+0j) shape: ", complex_.shape)

    txrx_frame_collation = np.concatenate((txrx_frame_collation, complex_), axis=3)

    print("TxRx Collation with Chirp and Frame Zeroes shape: ", txrx_frame_collation.shape)

    data_dict['txrx_frame_collation'] = txrx_frame_collation

    return data_dict

    chirp_diff = np.copy(master_frames)

    chirp_diff = np.diff(chirp_diff, axis=3)

    collate_chiprs_diff = np.copy(chirp_diff).reshape(-1, chirp_diff.shape[1], chirp_diff.shape[2], chirp_diff.shape[3]*chirp_diff.shape[4])

    txrx_frame_collation_diff = np.transpose(collate_chiprs_diff, (1,2,0,3))

    print("TxRx Collation Chirp Diff shape: ", txrx_frame_collation_diff.shape)

    frame_diff = np.copy(master_frames)

    frame_diff = np.diff(frame_diff, axis=0)

    frame_collate_diff = np.copy(frame_diff).reshape(-1, frame_diff.shape[1],frame_diff.shape[2], frame_diff.shape[3]*frame_diff.shape[4])

    frame_collate_diff = np.transpose(frame_collate_diff, (1,2,0,3))
    
    print("Frame Diff TxRx Collation shape: ", frame_collate_diff.shape)


    return data_dict

def get_stat_string(data,precision):
    return f"Max: {np.round(np.max(data),precision)} Min: {np.round(np.min(data),precision)} Mean: {np.round(np.mean(data),precision)} Std: {np.round(np.std(data),precision)}"

def cart2pol(cart:np.ndarray):
    return np.abs(cart), np.angle(cart)


def plot2d(original, filtered, dir, title_name, additional_stats=None):
    for i in tqdm(range(original.shape[0])):
        for j in range(original.shape[1]):
            fig, ax = plt.subplots(2,1, figsize=(11, 8))

            title = f"{title_name} TX {i+1} RX {j+1}"
            
            og_data = original[i,j,:,:]
            og_data = np.log(np.abs(og_data)+1e-6)
            og_data_stats = f"OG {get_stat_string(og_data, 4)}"

            filtered_data = filtered[i,j,:,:]
            filtered_data = np.log(np.abs(filtered_data)+1e-6)
            filtered_data_stats = f"LPF {get_stat_string(filtered_data, 4)}"

            ax[0].imshow(og_data, aspect='auto')
            ax[0].set_title(og_data_stats)
            fig.colorbar(ax[0].imshow(og_data, aspect='auto'), orientation='horizontal')

            ax[1].imshow(filtered_data, aspect='auto')
            ax[1].set_title(filtered_data_stats)
            fig.colorbar(ax[1].imshow(filtered_data, aspect='auto'), orientation='horizontal')

            fig.suptitle(title + f"\n{additional_stats}" if additional_stats else title)

            save_path = os.path.join(dir, f"{title}.png")
            plt.savefig(save_path)
            plt.close()

    
def plot3d(data,tx=0,rx=0):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X = np.arange(0, data.shape[3])
    Y = np.arange(0, data.shape[2])
    X, Y = np.meshgrid(X, Y)
    Z = np.log(np.abs(data[tx,rx,:,:])+1e-6)

    skip = 50

    X = X[::skip,::skip]
    Y = Y[::skip,::skip]
    Z = Z[::skip,::skip]
    ax.plot_surface(X, Y, Z, cmap='viridis')

    

    ax.set_title(f"TX {tx+1} RX {rx+1}")
    plt.show()

def plot_phase(signal, title, idx_1=0, idx_2=0):
    data = signal[idx_1, idx_2, :, :]

    a = f"Shape before: {data.shape}"

    data = data.flatten()

    b = f"Flattened shape: {data.shape}"
    
    angle = np.angle(data)
    angle = np.unwrap(angle)
    # angle = angle[::4096]

    # c = f"Shape after skipping: {angle.shape}"
    c=f""



    plt.plot(angle)
    plt.title(f"{title}\n{a}     {b}     {c}")
    plt.show()

    # exit(1)


def plots(data_dict):
    master_frames = data_dict['master_frames']
    txrx_frame_collation = data_dict['txrx_frame_collation']
    txrx_frame_collation_diff = data_dict['txrx_frame_collation_diff']
    frame_collate_diff = data_dict['frame_collate_diff']

    # filter_1 = np.copy(txrx_frame_collation) # Contains original raw data shape (12x4x511x4096)
    # filter_1 = filter_1[:,:,0:50,:] # Select the first 50 frames
    # filter_1 = np.mean(filter_1, axis=2) # Average over the frames
    # filter_1 = filter_1.reshape(filter_1.shape[0], filter_1.shape[1], 1, filter_1.shape[2]) # Reshape to (12x4x1x4096)
    # txrx_frame_filtered = np.copy(txrx_frame_collation) - filter_1 # Filtered data shape (12x4x511x4096)
    # txrx_frame_filtered = np.fft.fftshift(np.fft.fft(txrx_frame_filtered, axis=3), axes=3) # FFT of the filtered data over ADC samples
    # filter_1 = np.fft.fftshift(np.fft.fft(filter_1, axis=3), axes=3) # FFT of the filter over ADC samples
    txrx_frame_original = np.fft.fftshift(np.fft.fft(txrx_frame_collation, axis=3), axes=3) # FFT of the original raw data over ADC samples

    plot_phase(txrx_frame_original, "TxRx Collation")
    return

    filter_2 = np.copy(txrx_frame_collation_diff)
    filter_2 = np.mean(filter_2[:,:,0:50,:], axis=2)
    filter_2 = filter_2.reshape(filter_2.shape[0], filter_2.shape[1], 1, filter_2.shape[2])
    txrx_frame_filtered_diff = np.copy(txrx_frame_collation_diff) - filter_2
    txrx_frame_filtered_diff = np.fft.fftshift(np.fft.fft(txrx_frame_filtered_diff, axis=3), axes=3)
    txrx_frame_original_diff = np.fft.fftshift(np.fft.fft(txrx_frame_collation_diff, axis=3), axes=3)

    # plot_phase(txrx_frame_original_diff, "TxRx Collation Chirp Diff")

    filter_3 = np.copy(frame_collate_diff)
    filter_3 = np.mean(filter_3[:,:,0:50,:], axis=2)
    filter_3 = filter_3.reshape(filter_3.shape[0], filter_3.shape[1], 1, filter_3.shape[2])
    frame_filtered_diff = np.copy(frame_collate_diff) - filter_3
    frame_filtered_diff = np.fft.fftshift(np.fft.fft(frame_filtered_diff, axis=3), axes=3)
    frame_original_diff = np.fft.fftshift(np.fft.fft(frame_collate_diff, axis=3), axes=3)

    # plot_phase(frame_original_diff, "Frame Diff")

    base_dir = os.path.join(os.getcwd(), 'fft_plots', args.output_dir)

    txrx_dir = os.path.join(base_dir, 'txrx_collation')

    txrx_diff_dir = os.path.join(base_dir, 'txrx_collation_diff')

    frame_diff_dir = os.path.join(base_dir, 'frame_collation_diff')

    if not os.path.exists(txrx_dir):
        os.makedirs(txrx_dir,exist_ok=True)

    if not os.path.exists(txrx_diff_dir):
        os.makedirs(txrx_diff_dir,exist_ok=True)

    if not os.path.exists(frame_diff_dir):
        os.makedirs(frame_diff_dir,exist_ok=True)

    txrx_og = data_dict['txrx_frame_collation']
    txrx_diff_og = data_dict['txrx_frame_collation_diff']
    frame_og = data_dict['frame_collate_diff']

    txrx_og_stats = get_stat_string(txrx_og, 4)
    txrx_diff_og_stats = get_stat_string(txrx_diff_og, 4)
    frame_og_stats = get_stat_string(frame_og, 4)

    # plot2d(txrx_frame_original, txrx_frame_filtered, txrx_dir, "TxRx Collation", txrx_og_stats)
    # plot2d(txrx_frame_original_diff, txrx_frame_filtered_diff, txrx_diff_dir, "TxRx Collation Chirp Diff", txrx_diff_og_stats)
    # plot2d(frame_original_diff, frame_filtered_diff, frame_diff_dir, "Frame Diff", frame_og_stats)
    # plot3d(txrx_frame_original)

    # plot2d(txrx_frame_original, txrx_frame_filtered, txrx_dir, "TxRx Collation", None)
    # plot2d(txrx_frame_original_diff, txrx_frame_filtered_diff, txrx_diff_dir, "TxRx Collation Chirp Diff", None)
    # plot2d(frame_original_diff, frame_filtered_diff, frame_diff_dir, "Frame Diff", None)


    




    
if __name__ == "__main__":
    # Output directory that would hold the formatted data per frame
    OUTPUT_DIR: str = "output"

    # Number of samples
    NS: int = 256

    # Number of chirps
    NC: int = 64

    parser = argparse.ArgumentParser(
        prog="repack.py",
        description="MMWAVECAS-RF-EVM board recordings post-processing routine. "
                    "Repack the recordings into MIMO frames"
    )

    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for storing the mimo frames",
        type=str,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "-i", "--input-dir",
        help="Input directory containing the recordings",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.input_dir is None:
        print("[ERROR]: Missing input directory to read recordings")
        sys.exit(1)

    # The output directory will be created inside the data directory by default
    if (args.input_dir is not None) and (args.output_dir == OUTPUT_DIR):
        args.output_dir = os.path.join(args.input_dir, OUTPUT_DIR)

    # if not os.path.isdir(args.output_dir):
    #     os.makedirs(args.output_dir, exist_ok=True)

    # Load devices recording file paths
    master = load(args.input_dir, "master")
    slave1 = load(args.input_dir, "slave1")
    slave2 = load(args.input_dir, "slave2")
    slave3 = load(args.input_dir, "slave3")

    assert master != None, "Error with master data files"
    assert slave1 != None, "Error with slave1 data files"
    assert slave2 != None, "Error with slave2 data files"
    assert slave3 != None, "Error with slave3 data files"

    # Integrity status of the recordings
    # Check if the number of files generated for each device is
    # identical
    status: bool = True

    status = status and (len(master["data"]) == len(slave1["data"]))
    status = status and (len(master["data"]) == len(slave2["data"]))
    status = status and (len(master["data"]) == len(slave3["data"]))

    if not status:
        print("[ERROR]: Missing recording for cascade MIMO configuration")
        sys.exit(1)

    size: int = len(master["data"])

    # Number of frames recorded
    nf: int = 0

    # Number of frames generated from the last recording batch
    previous_nf: int = 0

    timestamps = np.array([])

    for idx in range(size):
        # Master file
        mf: str = master["data"][idx]
        mf_idx: str = master["idx"][idx]

        # Slave data files
        sf1: str = slave1["data"][idx]
        sf2: str = slave2["data"][idx]
        sf3: str = slave3["data"][idx]

        nf, _, timelogs = getInfo(mf_idx)

        # Skip if the number of valid frame is 0
        if not nf:
            continue

        timestamps = np.append(timestamps, timelogs)

        previous_nf = toframe(
            mf, sf1, sf2, sf3, # Input data files
            NS,
            NC,
            nf,
            args.output_dir,
            start_idx=previous_nf + 1
        )

        dict_data = reordering(previous_nf)

        lpf_dict = dict_data.copy()
        fft_dict = dict_data.copy()

        # # with open('data_dict.pkl', 'wb') as f:
        # #     pickle.dump(dict_data, f)

        # with open('data_dict.pkl', 'rb') as f:
        #     dict_data = pickle.load(f)

        plots(dict_data) 

