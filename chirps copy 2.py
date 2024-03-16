startFreq = 77*(10**9) # 77 GHz
freqSlope = 78.986*(10**6) # 78.986 MHz/us
idleTime = 5*(10**-6) # 5 us
adcStartTime = 6*(10**-6) # 6 us
adcSamples = 256
rampEndTime = 40*(10**-6) # 40 us
numChirps = 64
numLoops = 12
framePeriodicity = 100*(10**3) 
BW = 3159.44*(10**6) # 3159.44 MHz

config_dict = {
    "startFreq (GHz)": startFreq/10**9,
    "freqSlope (MHz/us)": freqSlope/10**6,
    "idleTime (s)": idleTime,
    "adcStartTime (s)": adcStartTime,
    "adcSamples": adcSamples,
    "rampEndTime (s)": rampEndTime,
    "numChirps": numChirps,
    "numLoops": numLoops,
    "framePeriodicity (s)": framePeriodicity/10**6,
    "BW (MHz)": BW/10**6
}

from pprint import pprint

pprint(config_dict,sort_dicts=False)

# Single Chirp cycle time = idleTime + adcStartTime + BW/slope
cycleTime = idleTime + adcStartTime + BW/freqSlope + 250*(10**-6) # 250 us for overhead

print("Single Chirp cycle time (BW/slope): ", cycleTime, "us")

print("Chirp Loop time (cycleTime*numChirps):", cycleTime*numChirps, "us")

print("Total Chirping time for frame (numLoops*cycleTime*numChirps):", cycleTime*numChirps*numLoops, "us")

print("Idle time between frames:", (framePeriodicity - cycleTime*numChirps*numLoops), "us")

print("Duty Cycle:~", (cycleTime*numChirps*numLoops)*100/framePeriodicity, "%")