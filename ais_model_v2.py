from matplotlib import pyplot as plt
from ais_variables import *
from ais_plotter import *
from neuron import h, gui
h.load_file("import3d.hoc")
import pylab as p
import numpy as np
import csv

# In the case of mechanisms not working - neuron.load_mechanisms('./channels')

class laminaNeuron():
    """A ball & stick neuron model describing """
    def __init__(self):
        self.label = 'passive'
        self.Rin = None
        self.Rheobase = None
        self.create_sections()
        self.build_topology()
        self.all = h.allsec()
        self.dendrites = [self.dend]
        self.neuron_cable = [self.soma, self.AIS, self.axon]
        self.define_geometry()
        self.set_biophysics()
        self.add_current_stim()
    def __repr__(self):
        return 'laminaNeuron'
    def load_morpholgoy():
        cell = h.Import3d_SWC_read()
        cell.input('dorsal_horn_network_project/cells/SWC_Files/c91662.swc')
        i3d = h.Import3d_GUI(cell, False)
        i3d.instantiate(self)
    def create_sections(self):
        """ Create morphological sections """
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.spacer = h.Section(name='spacer', cell=self)
        self.AIS = h.Section(name='AIS', cell=self)
        self.axon = h.Section(name='axon', cell=self)
    
    def build_topology(self):
        '''Connect sections together'''
        self.dend.connect(self.soma(0))
        self.spacer.connect(self.soma(1))
        self.AIS.connect(self.spacer(1))
        self.axon.connect(self.AIS(1))

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
        # Spacer
        self.spacer.L = SPACER_L        # change this for neuropathic conditions
        self.spacer.diam = SPACER_DIAM
        self.spacer.nseg = SPACER_NSEG
        # AIS
        self.AIS.L = AIS_L
        self.AIS.diam = AIS_DIAM
        self.AIS.nseg = AIS_NSEG
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
            sec.g_pas = GPAS  # in this version its .pas.g instead of g_pas
            sec.e_pas = E_PAS
        # Insert soma mechanisms:
        self.soma.insert('B_Na')
        self.soma.insert('KDRI')
        # Insert dendrite mechanisms
        self.dend.insert('SS')
        self.dend.insert('KDRI')
        # Insert axon initial segment mechanisms
        self.AIS.insert('B_Na')
        self.AIS.insert('KDRI')
        # Insert axon mechanisms
        self.axon.insert('B_Na')
        self.axon.insert('KDRI')
        # Set Active Reversal Potentials
        for sec in self.neuron_cable + self.dendrites:
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
        self.AIS.gnabar_B_Na = GNA_BAR_AIS
        self.AIS.gkbar_KDRI = GK_BAR_AIS
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
        self.AIS_v_vec = h.Vector()  # Membrane potential vector at dendrite
        self.t_vec = h.Vector()  # Time stamp vector
        self.soma_v_vec.record(self.soma(0.5)._ref_v)
        self.AIS_v_vec.record(self.AIS(0.5)._ref_v)
        self.t_vec.record(h._ref_t)
    
    # Figure out how to use properties
    # @property
    # def soma(self):
    #     return self.soma

    # @property
    # def dendrites(self):
    #     return self.dendrites


# class Simulation(laminaNeuron):
#     def __init__(self):
#         self.id = 0
#         self.rheobase = None
#         self.init_simulation()
#     def init_simulation(self, vinit=V_INIT, tstop=TSTOP, celsius=CELSIUS):
#         """Initialize and run a simulation.
#         :param celsius:
#         :param v_init:
#         :param tstop: Duration of the simulation.
#         """
#         h.v_init = vinit
#         h.tstop = tstop
#         h.celsius = celsius
#         h.dt = DT
#     def makeRecorders(self, segment, labels, rec=None):
#         if rec is None:
#             rec = {'t': h.Vector()}
#             rec['t'].record(h._ref_t)
#         for k,v in labels.items():
#             rec[k] = h.Vector()
#             rec[k].record(getattr(segment, v))
#         return rec

#     def makeIclamp(self, segment, dur, amp, delay=0):
#         stim = h.IClamp(segment)
#         stim.delay = delay
#         stim.dur = dur
#         stim.amp = amp
#         return stim

#     def rheobase_counter(self, cell, min_current=MIN_CURRENT, max_current=MAX_CURRENT, rheobase_step=RHEOBASE_STEP):
#         # Remove writing to file and put it seperately
#         self.id += 1
#         apc = h.APCount(cell.soma(0.5))
#         apc.thresh = THRESHOLD
#         current_lst = np.arange(min_current, max_current, rheobase_step)
#         for new_current in current_lst:
#             cell.stim.amp = new_current
#             h.run()
#             if apc.n > 0:
#                 print(apc.n)
#                 self.rheobase = new_current
#                 break
    
    
#     def get_if(self, segment, cur_inj_range, delay = DELAY, dur = DUR):
#         """
#         """
#         aps = []
#         ap = h.APCount(segment)
#         stim = self.makeIclamp(segment, dur, 0, delay)
#         for inj in cur_inj_range:
#             stim.amp = inj
#             h.run()
#             aps.append(ap.n)
#         return aps       
 
#     def calculate_input_resistance(self, segment, current_arr, delay=DELAY, dur=DUR, plot_flag=True):
#         """
#         Description: Extracts the input resistance from the neuron 

#         The protocol from the experiment was:
#         (RIN) was measured in voltage-clamp mode using negative 10- to 40-mV pulses from a holding level of –80 mV. 
#         Only cells with a resting potential (VR) negative to –60 mV were included into this study RIN of 1.7 GΩ
        
#         For the original mode the Rin was assumed to be around 1.7G, According to recorded cell which were 1.7 ± 0.3 GΩ
#         """
#         stim = self.makeIclamp(segment, dur, 0, delay)
#         rec = self.makeRecorders(segment, {'v': '_ref_v'})
#         ap = h.APCount(segment)
#         ap.thresh = -20
#         spks = h.Vector()
#         ap.record(spks)
#         I = []
#         V = []
#         if plot_flag:
#             p.figure()
#             p.subplot(1,2,1)
#         for k,i in enumerate(np.arange(current_arr[0],current_arr[1],current_arr[2])):     
#             spks.clear()
#             ap.n = 0
#             stim.amp = i
#             h.run()
#             spike_times = np.array(spks)
#             if len(np.intersect1d(np.nonzero(spike_times>delay)[0], np.nonzero(spike_times<delay+dur)[0])) == 0:
#                 # Recording of time calculated in ms   
#                 t = np.array(rec['t'])
#                 # Recording of voltage in mV
#                 v = np.array(rec['v'])
#                 # Extract steady state of the voltage
#                 idx = np.intersect1d(np.nonzero(t > delay+0.75*dur)[0], np.nonzero(t < delay+dur)[0])
#                 # Insert current
#                 I.append(i)
#                 # Calculate the mean of the steady state state, and substract the resting voltage pot
#                 V.append(np.mean(v[idx]) + 70) 
#             else:
#                 print('The neuron emitted spikes at I = %g pA' % (stim.amp*1e3))
#             if plot_flag:
#                 p.plot(1e-3*t,v)
#         #? Covert to microvolt, why ar we doing this? to help with the fit?        
#         V = np.array(V)*1e-3
#         # Convert current to pA units
#         I = np.array(I)*1e-9
#         #? Verify the polyfit function
#         poly = np.polyfit(I,V,1)
#         if plot_flag:
#             # Format the plot and plot the results
#             ymin,ymax = p.ylim()
#             p.plot([1e-3*(delay+0.75*dur),1e-3*(delay+0.75*dur)],[ymin,ymax],'r--')
#             p.plot([1e-3*(delay+dur),1e-3*(delay+dur)],[ymin,ymax],'r--')
#             p.xlabel('t (s)')
#             p.ylabel('V (mV)')
#             p.box(True)
#             p.grid(False)
#             p.subplot(1,2,2)
#             # Plots the current injected
#             x = np.linspace(I[0],I[-1],100)
#             y = np.polyval(poly,x)
#             p.plot(1e12*x,1e3*y,'k--')
#             p.plot(1e12*I,1e3*V,'bo')
#             p.xlabel('I (pA)')
#             p.ylabel('V (mV)')
#             # Save the figures
#             plt.savefig('sim_cell_figures/IV_Rin_Protocol.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)
#             plt.savefig('sim_cell_figures/IV_Rin_Protocol.png', format = 'png', bbox_inches = 'tight', dpi = 1200)
#             p.show()
#         #Convert to MegaOhm
#         Rin = poly[0]*1e-6
#         # Save the data
#         np.save('sim_cell_figures/IV_data', [V, I])
#         return Rin
    

#     def voltage_trace(self, cell):
#         apc = h.APCount(cell.soma(RECORDING_LOCATION))
#         apc.thresh = THRESHOLD
#         # laminaNeuron.set_recording()
#         # Setup Graphs:
#         plt.figure(figsize=(20, 8))
#         step = 0.04
#         num_steps = 2
#         for new_current in np.linspace(step, step * num_steps, num_steps):
#             cell.stim.amp = new_current 
#             h.run()
#             plt.plot(cell.t_vec, cell.soma_v_vec, label=str(new_current) + ', Spike Number = {}'.format(apc.n))
#         # Design Graph
#         plt.suptitle('Spike Graph', fontsize=14, fontweight='bold')
#         # pyplot.text(0.1, 2.8, "The number of action potentials is {}".format(apc.n))
#         plt.xlabel('time (ms)')
#         plt.ylabel('mV')
#         plt.legend()
#         plt.show()
         

#     def rheobase_protocol(self, cell):
#         apc = h.APCount(cell.soma(0.5))
#         apc.thresh = 0
#         current_lst = np.arange(MIN_CURRENT, MAX_CURRENT, RHEOBASE_STEP)
#         for new_current in current_lst:
#             cell.stim.amp = new_current
#             h.run()
#             if apc.n > 0:
#                 pyplot.plot(cell.t_vec, cell.soma_v_vec, label=str(new_current) + ', Rheobase = {}'.format(new_current))
#                 pyplot.suptitle('Spike Graph', fontsize=14, fontweight='bold')
#                 # pyplot.text(0.1, 2.8, "The number of action potentials is {}".format(apc.n))
#                 pyplot.xlabel('time (ms)')
#                 pyplot.ylabel('mV')
#                 pyplot.legend()
#                 pyplot.show()
#                 self.rheobase = new_current
#                 break
    
#     def plot_if(self, current_vector, freq_vector, color_vec=None):
#         """
#         Description: Plots the IF curves of 
#         """       
#         plt.figure()
#         ax1 = plt.subplot(1,1,1)
#         ax1.plot(current_vector, freq_vector)
#         # ax1.plot(current_vector, freq_onset_vector, '--', color=color_vec[1][0], label = currlabel + " onset rate")
#         plt.xlabel("Current [nA]")
#         plt.ylabel("Frequency [Hz]")
#         plt.title("I/F")
#         lg = plt.legend()
#         lg.get_frame().set_linewidth(0.5)
#         plt.show()
        

#     def plot_ais_plasticity_change(self, cell):
#         time_arr = []
#         soma_v_arr = []
#         rheobase_arr = []
#         for spacer_leng in SPACER_ARR:
#             # Set up simulation
#             cell.spacer.L = spacer_leng
#             # Append values
#             self.rheobase_counter(cell)
#             rheobase_arr.append(self.rheobase)
#             time_arr.append(np.array(cell.t_vec))
#             soma_v_arr.append(np.array(cell.soma_v_vec))
#         my_plotter(cell, time_arr, soma_v_arr, SPACER_ARR, rheobase_arr)

#     def extract_dv_dt(self, vtrace):
#         '''2-point first order finite difference to estimate dV/dt '''
#         dt = DT
#         dv = []
#         for i in range(1, len(vtrace)-2): 
#             dv.append((vtrace[i+1]-vtrace[i-1])/(2*dt))
#         return dv
    
#     def plot_phase_plane_trace(self, cell, stim_list):
#         # Make stim object       
#         stim = self.makeIclamp(cell.soma(0.5), DUR, 0, DELAY)
#         # Plot the phase plane of the graphs
#         fig1 = p.figure(1)
#         fig1_1 = fig1.add_subplot(221)
#         fig1_2 = fig1.add_subplot(223)
#         colors=["orangered","darkred","gold"]
#         for index, item in enumerate(stim_list):
#             print("Iinj =", item, "nA")
#             stim.amp = item
#             h.run()    
#             time = cell.t_vec
#             vtrace=np.array(cell.soma_v_vec).flatten()
#             vtraceAIS=np.array(cell.AIS_v_vec).flatten()
#             fig1_1.plot(time, vtrace, color=colors[index])
#             fig1_1.plot(time, vtraceAIS, color =colors[index], linestyle="dashed")
#             dv= self.extract_dv_dt(vtrace) 
#             dvAIS= self.extract_dv_dt(vtraceAIS)          
#             fig1_2.plot(vtrace[: len(dv)], dv, color=colors[index])
#             fig1_2.plot(vtraceAIS[: len(dvAIS)], dvAIS,linestyle="dashed", color=colors[index])  
        
#         # Edit the files
#         fsize = 10
#         fig1_1.set_xlabel("Time [ms]", fontsize=fsize)
#         fig1_1.set_ylabel("Voltage [mV]", fontsize=fsize)
#         fig1_2.set_xlabel("Voltage [mV]", fontsize=fsize)    
#         fig1_2.set_ylabel("dV/dt [V/s]", fontsize=fsize)

#         # Format the graph
#         #format the plot    
#         p.figure(1)
#         fig1_1.set_title("Iinj = 0.5 nA (yellow), 0.8 nA (red), 1.3 nA (orange) \n Soma (solid) and AIS (dashed)")
#         p.show()   

#     def save_simulation(self, laminaNeuron, file_name):
#         list_of_elem = [self.id, 1 / laminaNeuron.dend.g_pas,
#                           laminaNeuron.dend.Ra, laminaNeuron.dend.cm, laminaNeuron.dend.L,
#                           laminaNeuron.spacer.L, 1 / laminaNeuron.spacer.g_pas, laminaNeuron.spacer.Ra,
#                           laminaNeuron.spacer.cm, 1 / laminaNeuron.AIS.g_pas, self.rheobase]
#         with open(file_name, 'ab') as csvfile:
#             writer = csv.writer(csvfile, delimiter=',')
#             writer.writerow(list_of_elem)


# if __name__ == "__main__":
#     cell = laminaNeuron()
#     cell.set_recording()
#     sim = Simulation()
    
#     # Similar to Melnick experiment
#     currlabel = "Current ais model"
#     current_inj_current = np.linspace(0, 0.15, 9)
#     aps = sim.get_if(cell.soma(0.5), current_inj_current, DUR, DELAY)
#     # Plot the IF curve
#     sim.plot_if(current_inj_current, aps)

#     # Make sure it's the right units for input resistance
#     Rin = sim.calculate_input_resistance(cell.soma(0.5), [-0.3,-0.12,0.04], DELAY, DUR, False)
#     print(Rin)
    # sim.plot_phase_plane_trace(cell, [1.2])

    # Loading specific data: 
    # V,I = np.load('/home/michael/Desktop/DORSAL_HORN_PROJECT_FOLDER/NetPyNE_Project/cells/sim_cell_figures/IV_data.npy')
    # print(I)
