from matplotlib import pyplot as plt
from melnick_variables_original import * 
from ais_plotter import *
from ais_simulation import *
from neuron import h, gui
import numpy as np
import csv

class MelnickNeuron():
    """A ball & stick neuron model describing """
    def __init__(self):
        self.label = 'original_melnick_model'
        self.create_sections()
        self.build_topology()
        self.all = h.allsec()
        self.dendrites = [self.dend]
        self.neuron_cable = [self.soma, self.hillock, self.axon]
        self.define_geometry()
        self.set_biophysics()
        self.add_current_stim()
    def __repr__(self):
        return 'laminaNeuron'
    def create_sections(self):
        """ Create morphological sections """
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.hillock = h.Section(name='hillock', cell=self)
        self.axon = h.Section(name='axon', cell=self)
    def build_topology(self):
        '''Connect sections together'''
        self.dend.connect(self.soma(0))
        self.hillock.connect(self.soma(1))
        self.axon.connect(self.hillock(1))
    def define_geometry(self):
        '''Define Length, Diamter and        Number of Segment per Section'''
        # Dendrites
        self.dend.L = DEND_L
        self.dend.diam = DEND_DIAM
        self.dend.nseg = DEND_NSEG
        # Soma
        self.soma.L = SOMA_L
        self.soma.diam = SOMA_DIAM
        self.soma.nseg = SOMA_NSEG
        # hillock
        self.hillock.L = HILLOCK_L
        self.hillock.nseg = HILLOCK_NSEG
        # self.hillock.diam = HILLOCK_DIAM
        for seg, diam in zip(self.hillock, np.linspace(HILLOCK_BASE_DIAM, HILLOCK_END_DIAM, self.hillock.nseg)):
            seg.diam = diam
        # Axon
        self.axon.L = AXON_L
        self.axon.diam = AXON_DIAM
        self.axon.nseg = AXON_NSEG

    def set_biophysics(self):
        '''Set cell biophyisics including passive and active properties '''
        # Set passive membrane biophysics
        for sec in self.all:
            sec.Ra = R_A
            sec.cm = C_M
        # Set leaky channels for the model:
        for sec in self.all:
            sec.insert('pas')
            sec.g_pas = GPAS 
            sec.e_pas = E_PAS
        # Insert soma mechanisms:
        self.soma.insert('B_Na')
        self.soma.insert('KDRI')
        # Insert dendrite mechanisms
        self.dend.insert('SS')
        self.dend.insert('KDRI')
        # Insert axon initial segment mechanisms
        self.hillock.insert('B_Na')
        self.hillock.insert('KDRI')
        # Insert axon mechanisms
        self.axon.insert('B_Na')
        self.axon.insert('KDRI')
        # Set Active Reversal Potentials
        for sec in self.all:
            sec.ena = E_NA
            sec.ek = E_K
        # Set channel densities:
        # Dendrites
        self.dend.gnabar_SS = 0
        self.dend.gkbar_KDRI = GK_BAR_DEND
        # Soma
        self.soma.gnabar_B_Na = GNA_BAR_SOMA
        self.soma.gkbar_KDRI = GKDR_BAR_SOMA
        # AIS
        self.hillock.gnabar_B_Na = GNA_BAR_HILLOCK
        self.hillock.gkbar_KDRI = GK_BAR_HILLOCk
        # axon
        self.axon.gnabar_B_Na = GNA_BAR_AXON
        self.axon.gkbar_KDRI = GK_BAR_AXON

    def add_current_stim(self, delay=DELAY, dur=DUR, current=CURRENT, loc=RECORDING_LOCATION):
        """Attach a current Clamp to a cell.
        :param cell: Cell object to attach the current clamp.
        :param delay: Onset of the injected current.
        :param dur: Duration of the stimulus.
        :param amp: Magnitude of the current.
        :param loc: Location on the dendrite where the stimulus is placed.
        """
        self.stim = h.IClamp(self.soma(loc))
        self.stim.amp = current
        self.stim.delay = delay
        self.stim.dur = dur

    def set_recording(self):
        """Set soma, axon initial segment, and time recording vectors on the cell.
        :param cell: Cell to record from.
        :return: the soma, dendrite, and time vectors as a tuple.
        """
        self.soma_v_vec = h.Vector()  # Membrane potential vector at soma
        self.hillock_v_vec = h.Vector()  # Membrane potential vector at dendrite
        self.t_vec = h.Vector()  # Time stamp vector
        self.soma_v_vec.record(self.soma(0.5)._ref_v)
        self.hillock_v_vec.record(self.hillock(0.5)._ref_v)
        self.t_vec.record(h._ref_t)


# def simulation_gui():
#     # Add cell to each of the components
#     #Contorl Panel
#     sim_control = h.HBox()
#     sim_control.intercept(1)
#     h.nrncontrolmenu()
#     # attach_current_clamp(cell)
#     h.xpanel('TEST')
#     h.xlabel('Choose a  simulation to run')
#     h.xbutton('Spike Protocol',(sim.voltage_trace, cell))
#     h.xbutton('Rheobase Protocol',(sim.rheobase_protocol, cell))
#     h.xpanel()
#     #Output panel
#     g = h.Graph()
#     g.addvar('soma(0.5).v', cell.soma(0.5)._ref_v)
#     g.size(0, 1000, -90, 90)
#     h.graphList[0].append(g)
#     h.MenuExplore()
#     sim_control.intercept(0)
#     sim_control.map()
#     input()


# if __name__ == "__main__":
#     cell = laminaNeuron()
#     cell.set_recording()
#     sim = Simulation()
#     # simulation_gui()