import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import calendar, time, math
import sys

sys.path.append('..dorsal_horn_network_project/cells') 

def matrix_visualization(rheobase_mat):
    # print('a')
    array_of_slopes = []
    for array_of_data in rheobase_mat:
        result = np.polyfit(list(range(12)), list(array_of_data), 1)
        slope = result[-2]
        array_of_slopes.append(slope)
    # print(array_of_slopes)
    fig, ax = plt.subplots()
    # c = ax.pcolor(array_of_slopes)
    # ax.set_title('default: no edges')
    cax = ax.matshow(np.atleast_2d(array_of_slopes), interpolation=None)
    # for i in range(12):
    #     c = array_of_slopes[i]
    #     ax.text(i, 1, str(c), va='center', ha='center')
    #     # for j in range(15):
    fig.colorbar(cax)
    plt.show()

def visualize_plasticity_accross_model(data, SPACE_ARR):
    # Reshape the data
    data = np.array(data)
    data = data.T
    
    # Dictate columns and rows properties
    columns = ('Passive', 'Active', 'Passive Cone', 'Active Cone')
    rows = ['%d um' % x for x in SPACE_ARR[::-1]]
    colors = plt.cm.OrRd(np.linspace(0, 1, len(rows)))
    n_rows = len(data)
    n_columns = len(columns)
    index = np.arange(n_columns)
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(n_columns)
    
    # Plot bars and create text labels for the table
    cell_text = []
    
    # For column in range(n_columns):
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % x for x in y_offset])
        
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()
    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        rowColours=colors,
                        colLabels=columns,
                        loc='bottom')
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.35, bottom=0.35)
    # plt.ylabel("Rheobase increase with plasticity (pA) ")
    plt.ylabel("Rheobase increase with plasticity (pA) ")
    plt.xticks([])
    plt.title('AIS Plasticity vs Rheobase increase')
    plt.savefig('cell_data/figures/ais_summary_plot_normalized.svg', format = 'svg', bbox_inches = 'tight', dpi = 300)
    plt.savefig('cell_data/figures/ais_summary_plot_normalized.png', format = 'png', bbox_inches = 'tight', dpi = 300)
    plt.show()


def visualize_plasticity_not_normalized(data, SPACE_ARR):
    # Reshape the data
    data = np.array(data)
    data = data.T
    
    # Dictate columns and rows properties
    columns = ('Passive', 'Active', 'Passive Cone', 'Active Cone')
    rows = ['%d um' % x for x in SPACE_ARR[::-1]]
    colors = plt.cm.OrRd(np.linspace(0, 1, len(rows)))
    n_rows = len(data)
    n_columns = len(columns)
    index = np.arange(n_columns)
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(n_columns)
    
    # Plot bars and create text labels for the table
    cell_text = []
    
    # For column in range(n_columns):
    for row in range(n_rows-1,-1,-1):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        # plt.bar(index, data_reversed, bar_width, color=colors[row])
        # y_offset = y_offset + data[row]
        cell_text.append(['{:.2f}'.format(x) for x in data[row]])
        
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    # cell_text.reverse()
    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        rowColours=colors,
                        colLabels=columns,
                        loc='bottom')
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.35, bottom=0.35)
    # plt.ylabel("Rheobase increase with plasticity (pA) ")
    plt.ylabel("Rheobase increase with plasticity (pA) ")
    # plt.yticks(data, [val*1e3 for val in data], rotation=45)  # Set text labels and properties.
    plt.ylim(0.55,1.1)
    plt.xticks([])
    plt.title('AIS Plasticity vs Rheobase increase')
    plt.savefig('cell_data/figures/ais_summary_plot.svg', format = 'svg', bbox_inches = 'tight', dpi = 300)
    plt.savefig('cell_data/figures/ais_summary_plot.png', format = 'png', bbox_inches = 'tight', dpi = 300)
    plt.show()



def my_plotter(cell, data_x, data_y, data_header, data_legend):
    """
    Note this function normalizes the value on the axis, that's why it shows from 0 to 1
    """
    '''Dynamic plotter'''
    def do_plot(ax, data_x_row, data_y_row, data_header_item, data_legend_item):
        ax.plot(data_x_row, data_y_row, color='m')
        ax.set_title("AIS Distance(mu): {}".format(data_header_item),fontsize=12)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Potential (mV)')
        ax.axhline(data_legend[i], label="Rheobase: {}".format(data_legend_item), color="r")
        ax.legend(prop={'size': 6})
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    N = len(data_x)
    cols = 4
    rows = int(math.ceil(N / (cols - 1)))
    k = 0
    gs = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.7, width_ratios=[1]*(cols-1)+[5])

    #Add component plot
    fig = plt.figure(figsize=(16, 10))
    for i in range(cols):
        for j in range(rows):
            if k<=len(data_x)-1:
                ax = fig.add_subplot(gs[j, i])
                do_plot(ax, data_x[k], data_y[k], data_header[k], data_legend[k])
                k+=1

    #Add Summarizing Plot
    new_ax = fig.add_subplot(gs[0:, -1])
    # Normalize the data
    min_val = min(data_legend)
    max_val = max(data_legend)
    data_legend = [(x - min_val)/(max_val - min_val) for x in data_legend]
    new_ax.scatter(data_header, data_legend)
    new_ax.plot(data_header, data_legend)
    new_ax.set_title("Rheobase (pA) vs AIS Distance(mu):", fontsize=16)
    new_ax.set_xlabel('AIS Distance (mu)')
    new_ax.set_ylabel('Rheobase (pA)')
    for x, y in zip(data_header, data_legend):
        if y is None:
            continue
        label = "({:.2f},{:.2f})".format(x, y)
        new_ax.annotate(label,  # this is the text
                        (x, y),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center', fontsize=6)  # horizontal alignment can be left, right or center
    txt = "{} Cell Model: Dendrite Length {}, Dendrite Ra {}, Spacer Ra {}, Dendrite Rm {}, Spacer Rm {} Dendrite Cm {}, Spacer Cm {}, Dendrite g_pas {}, Spacer g_pas {} "
    plt.figtext(0.5, 0.01, txt.format(cell.label, cell.dend.L, cell.dend.Ra, cell.spacer.Ra, 1/cell.dend.g_pas, 1/cell.spacer.g_pas, cell.dend.cm, cell.spacer.cm, cell.dend.g_pas, cell.spacer.g_pas), wrap=True, horizontalalignment='center',
                   fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    # Save graph
    gmt = time.gmtime()
    # ts stores timestamp
    ts = calendar.timegm(gmt)
    plt.savefig('cell_data/figures/{}_simulation_plot_{}.png'.format(cell.label ,str(ts)), dpi=300)
    plt.show()

def summary_plotter(data_x, data_y, data_header):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(data_x[i], data_y[i], color='m')
        ax.set_title("AIS Distance(mu): {}".format(data_header[i]))
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Potential (mV)')
        ax.legend(prop={'size': 6})

def correlation_plotter(cell, data_x, data_y, x_str):
    Coeffecient = np.corrcoef(data_x, data_y)
    plt.scatter(data_x, data_y) 
    plt.title('A plot to show the correlation between variables spacer distance and {}'.format(x_str))
    plt.ylabel(x_str)
    plt.xlabel('Spacer distance (mu)')
    plt.plot(np.unique(data_x), np.poly1d(np.polyfit(data_x, data_y, 1))(np.unique(data_x)), color='blue')
    plt.savefig('cell_data/figures/{}_correlation_plot_{}.png'.format(cell.label,x_str))
    plt.show()
