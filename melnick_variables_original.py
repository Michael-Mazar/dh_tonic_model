"""
All diameter and length parameters are in um units
"""
#---------------------------- Morphology --------------------------#
# Neuron dendrite morphology:
DEND_L = 1371           # [um] in units
DEND_DIAM = 1.4         # [um] in units
DEND_NSEG = 21

# Neuron soma morphology:
SOMA_DIAM = 10
SOMA_L = 10
SOMA_NSEG = 11

# Axon initial segment morphology:
HILLOCK_BASE_DIAM = 1
HILLOCK_END_DIAM = 0.5
HILLOCK_L = 30              # (ADDED) Adapted from immunohistochemical staining results
HILLOCK_NSEG = 31           #

# Axon morphology: Sealed end
AXON_L = 0.001
AXON_DIAM = 0.001
AXON_NSEG = 51

#---------------------- Reversal Potential --------------------------#
E_NA = 60
E_K = -84
E_PAS = -70

#---------------------- Passive Properties --------------------------#
R_A = 80                    # [Ohm*cm]
C_M = 1                     # [uf/cm^2]
GPAS = 1.1 * pow(10, -5)

#---------------------- Active Properties  --------------------------#
# Dendrite active properties:
GNA_BAR_DEND = 0            # S/cm^2 in units
GK_BAR_DEND = 0.034         # S/cm^2 in units

# Soma active properties
GNA_BAR_SOMA = 0.008
GKDR_BAR_SOMA = 0.0043

# Axon initial segment active properties
GNA_BAR_HILLOCK = 1.8         # Adjusted from Hillock model, increased by 2
GK_BAR_HILLOCk = 0.076

# Axon active properties
GNA_BAR_AXON = 0
GK_BAR_AXON = 0

#---------------------- Simulation Control  --------------------------#
STEPS_PER_MS = 20           # 0.02
TSTOP = 1200
V_INIT = -70
DT = 0.05
CELSIUS = 23
THRESHOLD = -30             # (CHANGED) Adjusted from Hillock model, increased by 2
RECORDING_LOCATION = 0.5
DELAY = 500
DUR = 500
CURRENT = 0 #0.02
MIN_CURRENT = 0.01
MAX_CURRENT = 0.20
RHEOBASE_STEP = 0.0005
