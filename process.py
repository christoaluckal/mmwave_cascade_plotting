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


class MMWProcessor:
    def __init__(self, frames) -> None:
        self.master_frames = frames

    def convertToComplex(self, frames: np.ndarray = None) -> np.ndarray:
        if frames is None:
            frames = self.master_frames

        complex_frames = frames[...,0]+1j*frames[...,1]
        return complex_frames
    
    def getCollatedFrames(self, frames: np.ndarray) -> np.ndarray:
        collate_chiprs = np.copy(frames).reshape(-1, frames.shape[1], frames.shape[2], frames.shape[3]*frames.shape[4])
        txrx_frame_collation = np.transpose(collate_chiprs, (1,2,0,3))
        print("TxRx Collation shape: ", txrx_frame_collation.shape)

        return txrx_frame_collation
    
    def getCollatedChirpDiff(self, frames: np.ndarray) -> np.ndarray:
        chirp_diff = np.copy(frames)
        chirp_diff = np.diff(chirp_diff, axis=3)
        collate_chiprs_diff = np.copy(chirp_diff).reshape(-1, chirp_diff.shape[1], chirp_diff.shape[2], chirp_diff.shape[3]*chirp_diff.shape[4])
        txrx_frame_collation_diff = np.transpose(collate_chiprs_diff, (1,2,0,3))
        print("TxRx Collation Chirp Diff shape: ", txrx_frame_collation_diff.shape)

        return txrx_frame_collation_diff
    
    def getCollatedFrameDiff(self, frames: np.ndarray) -> np.ndarray:
        frame_diff = np.copy(frames)
        frame_diff = np.diff(frame_diff, axis=0)
        frame_collate_diff = np.copy(frame_diff).reshape(-1, frame_diff.shape[1],frame_diff.shape[2], frame_diff.shape[3]*frame_diff.shape[4])
        frame_collate_diff = np.transpose(frame_collate_diff, (1,2,0,3))
        print("Frame Diff TxRx Collation shape: ", frame_collate_diff.shape)

        return frame_collate_diff
    
    def padChirpsData(self, frames: np.ndarray, chirpPads: int = 152, pad_type="complex") -> np.ndarray:
        if pad_type == "complex":
            zeroes = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3], chirpPads))
            ones = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3], chirpPads))
            complex_ = ones+1j*zeroes
            frames = np.concatenate((frames, complex_), axis=4)

            print("Padded Complex Chirps shape: ", frames.shape)

        else:
            zeroes = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3], chirpPads))
            frames = np.concatenate((frames, zeroes), axis=4)
            print("Padded Frames shape: ", frames.shape)

        return frames
    
    def converToFrameFormat(self, frames: np.ndarray, frame_pad: int = 25446) -> np.ndarray:
        n_frames = frames.shape[0]
        numLoops = frames.shape[1]
        numDevices = frames.shape[2]
        numChirps = frames.shape[3]
        numSamples = frames.shape[4]

        device_primary = np.transpose(frames,(2,0,1,3,4))
        device_primary = device_primary.reshape(numDevices, n_frames, numLoops*numChirps*numSamples)

        frame_pad = np.zeros((numDevices, n_frames, frame_pad))

        concatenated = np.concatenate((device_primary, frame_pad), axis=2)
        print("Frame format shape: ", concatenated.shape)



if __name__ == "__main__":
    from binreader import MMWBinReader

    parser = argparse.ArgumentParser(description="Process MMWCAS-RF-EVM & MMWCAS-DSP-EVM kit Recordings.")
    parser.add_argument("--input", type=str, help="Path to the input directory")

    args = parser.parse_args()

    reader = MMWBinReader()
    data = reader.readBinFromPath(args.input, 256, 64)
    print(data.shape)

    processor = MMWProcessor(data)
    complex_frames = processor.convertToComplex()

    collated_padded = processor.padChirpsData(complex_frames, chirpPads=0)

    processor.converToFrameFormat(collated_padded,frame_pad=0)


    

