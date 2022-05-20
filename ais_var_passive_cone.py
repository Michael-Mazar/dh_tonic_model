"""
All diameter and length parameters are in um units
"""
#---------------------------- Morphology --------------------------#
# Neuron dendrite morphology:
DEND_L = 500 #1371           # [um] in units
DEND_DIAM = 0.86 #1.4         # [um] in units
DEND_NSEG = 20

# Neuron soma morphology:
SOMA_DIAM = 10
SOMA_L = 10
SOMA_NSEG = 10

# Axon initial segment morphology:
SPACER_DIAM = 3
SPACER_L = 5              # (ADDED) Adapted from immunohistochemical staining results
SPACER_NSEG = 31           #

# Axon initial segment morphology:
AIS_DIAM = 1.3          # Continuous cylinder diameter
AIS_L = 25 # AIS_L = 15              # (ADDED) Adapted from immunohistochemical staining results
AIS_NSEG = 10           #

# Axon morphology: Sealed end
AXON_L = 0.001
AXON_DIAM = 0.001
AXON_NSEG = 10

#---------------------- Reversal Potential --------------------------#
E_NA = 60
E_K = -84
E_PAS = -70

#---------------------- Passive Properties --------------------------#
R_A = 80                    # [Ohm*cm]
C_M = 1                     # [uf/cm^2]
GPAS_NEW = 5 * pow(10, -5) # (CHANGED) [S/cm2] Adjusted to prevent spontaneous firing

#---------------------- Active Properties  --------------------------#
# Dendrite active properties:
GNA_BAR_DEND = 0            # S/cm^2 in units
GK_BAR_DEND = 0.034         # S/cm^2 in units

# Soma active properties
GNA_BAR_SOMA = 0.008
GKDR_BAR_SOMA = 0.0043

# Axon initial segment active properties
GNA_BAR_AIS = 1.8
GK_BAR_AIS = 0.076

# Axon active properties
GNA_BAR_AXON = 0
GK_BAR_AXON = 0

#---------------------- Simulation Control  --------------------------#
STEPS_PER_MS = 20           # 0.02
TSTOP = 1200
V_INIT = -70
DT = 0.05
CELSIUS = 23
THRESHOLD = -30             # (CHANGED) Adjusted from SPACER model, increased by 2
RECORDING_LOCATION = 0.5
DELAY = 500
DUR = 500 # DUR = 300 # DUR = 500
CURRENT = 0 #0.02
MIN_CURRENT = 0.0005
MAX_CURRENT = 0.06
RHEOBASE_STEP = 0.0005
SPACER_ARR = [2, 3, 5, 10, 20, 30, 40, 50, 70, 90, 100, 150]
