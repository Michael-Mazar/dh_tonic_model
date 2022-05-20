from matplotlib import pyplot as plt
from ais_model_v2 import laminaNeuron
from ais_active_model import ActivelaminaNeuron
from ais_cone_active_model import ActiveConelaminaNeuron
from ais_cone_passive_model import PassiveConelaminaNeuron
from ais_original_model import MelnickNeuron


# from ais_original_model import MelnickNeuron
from ais_simulation import *
from neuron import h, gui

sys.path.append('..dorsal_horn_network_project/cells')
sys.path.append('..dorsal_horn_network_project/cells/cell_data/numerical_data')


class MyWindow:
    def __init__(self, cell):
        self.cell = cell
        self.cell.set_recording()
        self.passive_selected = 0
        self.active_selected = 0
        self.passive_cone_selected = 0
        self.active_cone_selected = 0
        self.test = 0
        self.open_window()
        # input()
    
    def open_window(self):
        #Contorl Panel
        self.sim_control = h.HBox()
        self.sim_control.intercept(1)
        h.nrncontrolmenu()
        # attach_current_clamp(cell)
        h.xpanel('Protocol Selection Window')
        h.xlabel(str(self.test))
        h.xlabel('Select a cell type to run simulation on')
        h.xradiobutton('Passive cell', (self.clicked, 1), self.passive_selected)
        h.xradiobutton('Active cell', (self.clicked, 2), self.active_selected)
        h.xradiobutton('Active cone cell', (self.clicked, 3), self.passive_cone_selected)
        h.xradiobutton('Passive cone cell', (self.clicked, 4), self.active_cone_selected)
        h.xlabel('Choose a simulation to run')
        h.xbutton('Plot & Extract Voltage Trace',(sim.get_voltage_trace, self.cell))
        h.xbutton('Plot & Extract Input Resistance',(sim.get_input_resistance, self.cell))
        h.xbutton('Plot & Extract Channel Conductances',(sim.get_channel_conductances, self.cell))
        h.xbutton('Plot & Extract Rheobase',(sim.rheobase_protocol, self.cell))
        h.xbutton('Plot & Extract Frequency-Input (FI) Curve',(sim.get_if, self.cell))
        h.xbutton('Plot & Extract Phase Plane Dynamics',(sim.get_phase_plane_trace, self.cell))
        h.xbutton('Examine AIS change effect on cell Rheobase',(sim.get_ais_distalization_effect, self.cell))
        h.xpanel()
        #Output panel
        g = h.Graph()
        g.addvar('soma(0.5).v', self.cell.soma(0.5)._ref_v)
        # g.addvar('AIS(0.5).v', cell.AIS(0.5)._ref_v)
        g.size(0, 1000, -90, 90)
        h.graphList[0].append(g)
        h.MenuExplore()
        self.sim_control.intercept(0)
        self.sim_control.map()   
    
    def clicked(self, choice):
        if choice == 1: 
            self.test = 1
            self.cell = laminaNeuron()
            self.cell.set_recording()
            print('Switched selected cell: {} cell'.format(self.cell.label)) 
            self.sim_control.unmap()
            # self.sim_control.map()
            self.open_window()
        if choice == 2: 
            self.sim_control.unmap()
            self.test = 2
            self.cell = ActivelaminaNeuron()
            self.cell.set_recording()
            print('Switched selected cell: {} cell'.format(self.cell.label))
            # self.sim_control.map()
            self.open_window()
            # self.sim_control.unmap()
            # h.xlabel(str(self.test))
            # self.sim_            self.test = 1control.map()
        if choice == 3: 
            self.sim_control.unmap()
            self.test = 3
            self.cell = ActiveConelaminaNeuron()
            print('Switched selected cell: active cone cell')
            self.open_window()
        if choice == 4: 
            self.test = 4
            self.cell = PassiveConelaminaNeuron()
            self.cell.set_recording()
            print('Switched selected cell: passive cone cell')
            self.open_window()
        
if __name__ == "__main__":  
    # Load data and analyze
    sim = Simulation()
    # rheobase_arr = np.load('temp_test.npz')
    # rheobase_arr = rheobase_arr['rheobase_mat']
    # matrix_visualization(rheobase_arr)
    #   
    
    ## Defined a neuron
    # cell = laminaNeuron()
    # cell = ActivelaminaNeuron()
    cell = ActiveConelaminaNeuron()
    # cell = PassiveConelaminaNeuron()
    # cell = MelnickNeuron()

    

    ## Create simulation variable
    window = MyWindow(cell)
    # param = 'cell.dend.L'
    # sim.get_plasticity_change_multi(active_cone_cell, param)

    # Run simulation for AIS: 
    # param = 'cell.AIS.L'
    # sim.get_plasticity_change_multi(cell, param)
    

    ## Calculate the input resistance
    # print(Rin)
    # Create output file
    # Load data
    # path = '/cell_data/numerical_data'
    # passive_rheobase_file = path+'laminaNeuronais_plasticity_20220122-134115.h5'
    # passive_cone_rheobase_file = path+'PassiveConelaminaNeuronais_plasticity_20220122-145852.h5'
    # active_rheobase_file = path+'ActiveConelaminaNeuronais_plasticity_20220122-152643.h5'
    # active_rheobase_cone_file = path+'ActivelaminaNeuronais_plasticity_20220122-141324.h5'
    # passive_data = sim.load_simulation(passive_rheobase_file)
    # passive_cone_data = sim.load_simulation(passive_cone_rheobase_file)
    # active_rheobase_data = sim.load_simulation(active_rheobase_file)
    # active_rheobase_cone_data = sim.load_simulation(active_rheobase_cone_file)
    # Import the os module
    # import os

    # # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    # # # Change the current working directory
    # os.chdir('dorsal_horn_network_project/cells')

    # # # Print the current working directory
    # print("Current working directory: {0}".format(os.getcwd()))
    
    ## Load from excel
    # rheobase_mat = []
    # rb_mat_not = []
    # labels = ['passive', 'active', 'passive_cone', 'active_cone']
    # for label in labels:
    #     df = pd.read_excel('cell_data/numerical_data/{}.xlsx'.format(label), sheet_name='Excitability Correlation')
    #     rb_mat_not.append([x*1e3 for x in df['Rheobase (nA)'].tolist()])
    #     rheobase_mat.append(sim.get_normalized_data(df['Rheobase (nA)'].tolist()))
    # visualize_plasticity_accross_model(rheobase_mat, SPACER_ARR)
    # visualize_plasticity_not_normalized(rb_mat_not, SPACER_ARR)   
    
    # passive_data = []

    # rheobase_mat = [passive_data, passive_cone_data, active_rheobase_data, active_rheobase_cone_data]
    # for row in range(len(rheobase_mat)):
    #     rheobase_mat[row] = sim.get_normalized_data(rheobase_mat[row])
    # visualize_plasticity_accross_model(rheobase_mat, SPACER_ARR)
   
    # Run the simulation GUI
    # simulation_gui(cell)