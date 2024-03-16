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


class MMWBinReader:
    def __init__(self) -> None:
        """Re-Format the raw radar ADC recording.

        The raw recording from each device is merge together to create
        separate recording frames corresponding to the MIMO configuration.

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
        self.nwave: int = 2

        # Number of cascaded radar chips
        self.nchip: int = 4

        # Number of TX antenna
        self.ntx: int = 3

        # Number of RX antenna per chip
        self.nrx: int = 4


    def getInfo(self,idx_file: str):
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


    def load(self,inputdir: str, device: str):
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


    def getFrames(self,
            mf: str, sf0: str, sf1: str, sf2: str,
            ns: int, nc: int, nf: int,
        ) -> np.ndarray:

        master_frames = []

        for fidx in range(nf):
            # Number of items to read (here items are 16-bit integer values)
            nitems: int = self.nwave * ns * nc * self.nrx * self.ntx * self.nchip
            # Offet to read the bytes of a given frame
            # The multiplication by "2" is to count for the size of 16-bit integers
            offset: int = fidx * nitems * 2

            dev1 = np.fromfile(mf, dtype=np.int16, count=nitems, offset=offset)
            dev2 = np.fromfile(sf0, dtype=np.int16, count=nitems, offset=offset)
            dev3 = np.fromfile(sf1, dtype=np.int16, count=nitems, offset=offset)
            dev4 = np.fromfile(sf2, dtype=np.int16, count=nitems, offset=offset)

            dev1 = dev1.reshape(nc, self.ntx * self.nchip, ns, self.nrx, 2)
            dev2 = dev2.reshape(nc, self.ntx * self.nchip, ns, self.nrx, 2)
            dev3 = dev3.reshape(nc, self.ntx * self.nchip, ns, self.nrx, 2)
            dev4 = dev4.reshape(nc, self.ntx * self.nchip, ns, self.nrx, 2)

            dev1 = np.transpose(dev1, (1, 3, 0, 2, 4))
            dev2 = np.transpose(dev2, (1, 3, 0, 2, 4))
            dev3 = np.transpose(dev3, (1, 3, 0, 2, 4))
            dev4 = np.transpose(dev4, (1, 3, 0, 2, 4))

            master_frames.append(dev1)

        

        return np.array(master_frames)
    
    def readBinFromPath(self, path: str, adcSamples: int, nChirps: int, nFrames: int = 0):
        # Number of samples
        NS: int = adcSamples

        # Number of chirps
        NC: int = nChirps

        # Load devices recording file paths
        master = self.load(path, "master")
        slave1 = self.load(path, "slave1")
        slave2 = self.load(path, "slave2")
        slave3 = self.load(path, "slave3")

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

        timestamps = np.array([])

        mf: str = master["data"][0]
        mf_idx: str = master["idx"][0]

        # Slave data files
        sf1: str = slave1["data"][0]
        sf2: str = slave2["data"][0]
        sf3: str = slave3["data"][0]

        nf, _, timelogs = self.getInfo(mf_idx)

        # Skip if the number of valid frame is 0
        if not nf:
            return None

        timestamps = np.append(timestamps, timelogs)

        master = self.getFrames(
            mf, sf1, sf2, sf3, # Input data files
            NS,
            NC,
            nf
        )

        return master




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the input directory")

    args = parser.parse_args()

    reader = MMWBinReader()
    data = reader.readBinFromPath(args.input, 256, 64)
    print(data.shape)

