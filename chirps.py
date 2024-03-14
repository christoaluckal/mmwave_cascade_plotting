startFreq = 77*1000 
freqSlope = 78.986
idleTime = 5
adcStartTime = 6
adcSamples = 256
rampEndTime = 40
numChirps = 64
numLoops = 12
framePeriodicity = 100*1000
BW = 3159.44 

config_dict = {
    "startFreq (GHz)": startFreq/1000,
    "freqSlope (MHz/us)": freqSlope,
    "idleTime (us)": idleTime,
    "adcStartTime (us)": adcStartTime,
    "adcSamples": adcSamples,
    "rampEndTime (us)": rampEndTime,
    "numChirps": numChirps,
    "numLoops": numLoops,
    "framePeriodicity (s)": framePeriodicity/1000000,
    "BW (MHz)": BW
}

from pprint import pprint

pprint(config_dict,sort_dicts=False)

singleChirpCycleTime = idleTime + adcStartTime + BW/freqSlope
print("\nsingleChirpCycleTime = idleTime + adcStartTime + BW/freqSlope = ", singleChirpCycleTime, "us")

chirpLoopTime = singleChirpCycleTime*numChirps
print("\nchirpLoopTime = singleChirpCycleTime*numChirps = ", chirpLoopTime, "us")

totalChirpingTime = chirpLoopTime*numLoops
print("\ntotalChirpingTime = chirpLoopTime*numLoops = ", totalChirpingTime, "us")

idleTimeBetweenFrames = framePeriodicity - totalChirpingTime
print("\nidleTimeBetweenFrames = framePeriodicity - totalChirpingTime = ", idleTimeBetweenFrames, "us")

dutyCycle = (totalChirpingTime)*100/framePeriodicity
print("\ndutyCycle = (totalChirpingTime)*100/framePeriodicity = ", dutyCycle, "%")

perLoopFrameADCsamples = adcSamples*numChirps
print("\nperLoopFrameADCsamples = adcSamples*numChirps = ", perLoopFrameADCsamples)

nonZeroValuesPerUs = perLoopFrameADCsamples/totalChirpingTime
print("\nnonZeroValuesPerUs = perLoopFrameADCsamples/totalChirpingTime = ", nonZeroValuesPerUs)

totalValuesPerChirp = int(framePeriodicity*nonZeroValuesPerUs)
print("\ntotalValuesPerChirp = floor(framePeriodicity*nonZeroValuesPerUs) = ", totalValuesPerChirp)

zeroesToAppend = totalValuesPerChirp - perLoopFrameADCsamples
print("\nzeroesToAppend = totalValuesPerChirp - frameADCsamples = ", zeroesToAppend)