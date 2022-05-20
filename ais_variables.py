"""
All diameter and length parameters are in um units
"""
#---------------------------- Morphology --------------------------#
# Neuron dendrite morphology:
DEND_L = 500 #1371 #500            # [um] in units, following cable equations adjusted from 500 um
DEND_DIAM =  0.86 #1.4 #0.86        # [um] in units, following cable equations
DEND_NSEG = 21

# Neuron soma morphology:
SOMA_DIAM = 10
SOMA_L = 10
SOMA_NSEG = 11

# Neuron space between soma and axon initial segment (AIS) morphology:
SPACER_DIAM = 1.3 #1.5       # Adjusted for Conductance and replacement of hillock
SPACER_L = 5            # (CHANGED) Hillock in the original was 3, adapted to immunohistochemical staining results
SPACER_NSEG = 51        # In the original model it was 30

# Axon initial segment morphology:
AIS_DIAM = 1.3 #1.5          # Adjusted for conductuctance and replacement of hillock 1.5
AIS_L = 25 # AIS_L = 15              # (ADDED) Adapted from immunohistochemical staining results, previously was 50
AIS_NSEG = 41           #

# Axon morphology: Sealed end
AXON_L = 0.001
AXON_DIAM = 0.001
AXON_NSEG = 11

#---------------------- Reversal Potential --------------------------#
E_NA = 60
E_K = -84
E_PAS = -70

#---------------------- Passive Properties --------------------------#
R_A = 80                    # [Ohm*cm]
C_M = 1                     # [uf/cm^2]
GPAS = 5 * pow(10, -5)

#---------------------- Active Properties  --------------------------#
# Dendrite active properties:
GNA_BAR_DEND = 0            # Units: S/cm^2 or ans*1e3 mmho/cm2 or  ans*1e4 (1e+12/1e8) pS/µm2 
GK_BAR_DEND = 0.034         # S/cm^2 in units

# Soma active properties
GNA_BAR_SOMA = 0.008
GKDR_BAR_SOMA = 0.0043

# Axon initial segment active properties
GNA_BAR_AIS = 1.8         # 
GK_BAR_AIS = 0.076        # 
# GNA_BAR_AIS_OLD = 1.8*2.5

# Axon active properties
GNA_BAR_AXON = 0
GK_BAR_AXON = 0

#---------------------- Simulation Control  --------------------------#
PARAM_SPACE = {'cell.dend.L' : [450, 650, 10],
                      'cell.AIS.L': [15, 35, 10],
                      'cell.dend.g_pas': [1e-6, 1e-5, 10],
                      'cell.AIS.gnabar_B_Na':[1, 3, 10],
                      'cell.AIS.gkbar_KDRI': [0.076, 1, 10],
                      'h.celsius': [20, 37, 10]
                      }

STEPS_PER_MS = 20           # 0.02
TSTOP = 1200
V_INIT = -70
DT = 0.025
CELSIUS = 23
THRESHOLD = -30             # (CHANGED) Adjusted from Hillock model, increased by 2
RECORDING_LOCATION = 0.5
DELAY = 500
DUR = 500 # DUR = 300 # Changed from DUR = 500
CURRENT = 0 #0.02
MIN_CURRENT = 0
MAX_CURRENT = 0.06
RHEOBASE_STEP = 0.00002
SPACER_ARR = list(range(1,27,2))
CUR_INJ_PROTOCOL = [0.05, 0.1, 0.15]
CUR_INJ_PROTOCOL_SMALL = [0.005, 0.01, 0.015]
# SPACER_ARR = [2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
