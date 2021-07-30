import tkinter as tk
from tkinter import * 
from tkinter import ttk

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('./matplotlib.mplstyle')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
import h5py
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

class MainWindow():
    def __init__(self):
        self.root_window = tk.Tk()
        self.root_window.title("Visualizer")
        self.init_window_size(width=1280, height=720) ## WXGA

        ## vars
        self.file = h5py.File('./../../../dataset_folder/synthetic_data.hdf5', 'r')
        self.maingroup = tk.StringVar()
        self.datagroup = tk.StringVar()
        self.data = tk.StringVar()
        self.ts = None

        ## binding keys to functions
        self.root_window.bind('<Return>',self.reset)

        self.create_statusbar()
        self.create_black_frame()
        self.create_labelframes()
        self.initiate_plotframe()
        self.initiate_optionsframe()
        self.init_plot()

    def init_window_size(self, width, height):
        screen_width = self.root_window.winfo_screenwidth()
        screen_height = self.root_window.winfo_screenheight()
        center_x = int(screen_width//2 - width//2)
        center_y = int(screen_height//2 - height//2)
        self.root_window.geometry(f'{width}x{height}+{center_x}+{center_y}')
        self.root_window.minsize(width,height)

    def create_statusbar(self):
        self.statusbar_note = tk.StringVar()
        self.statusbar = tk.Label(self.root_window, textvariable=self.statusbar_note, bd=0.5, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0))

    def create_black_frame(self):
        self.blackframe = ttk.Frame(self.root_window)
        self.blackframe.pack(expand=True, fill='both')
        self.blackframe.columnconfigure(0,weight=1)
        self.blackframe.columnconfigure(1,weight=7)
        self.blackframe.columnconfigure(2,weight=1)
        self.blackframe.rowconfigure(0, weight=1)

    def create_labelframes(self):
        ## tools_frame
        self.plotframe = ttk.LabelFrame(self.blackframe, text='Tools')
        self.plotframe.grid(column=0, row=0, padx=(10,10), pady=(10,0), sticky='nesw')
        self.plotframe.grid_propagate(False)
        ## plotframe
        self.plotframe = ttk.LabelFrame(self.blackframe, text='Plot')
        self.plotframe.grid(column=1, row=0, padx=(0,10), pady=(10,0), sticky='nesw')
        self.plotframe.grid_propagate(False)
        ## optionsframe
        self.optionsframe = ttk.LabelFrame(self.blackframe, text='Options')
        self.optionsframe.grid(column=2,row=0, padx=(0,10), pady=(10,0), sticky='nesw')
        self.optionsframe.grid_propagate(False)

    def initiate_plotframe(self):
        self.plotframe.columnconfigure(0, weight=1)
        self.plotframe.rowconfigure(0, weight=1)
        # self.plotframe.rowconfigure(1, weight=0)

        self.figureframe = tk.Frame(self.plotframe)
        self.figureframe.grid(column=0, row=0, sticky='nes')
        self.figureframe.grid_propagate(False)

        self.toolbarframe = tk.Frame(self.plotframe)
        self.toolbarframe.grid(column=0, row=0, sticky='ws')


    def initiate_optionsframe(self):
        self.optionsframe.columnconfigure(0,weight=1)
        ## main group selection
        self.optionsframe.rowconfigure(0,weight=0)
        self.optionsframe.rowconfigure(1,weight=0)
        ## dataset group selection
        self.optionsframe.rowconfigure(2,weight=0)
        self.optionsframe.rowconfigure(3,weight=0)
        ## dataset group selection
        self.optionsframe.rowconfigure(4,weight=0)
        self.optionsframe.rowconfigure(5,weight=0)
        ## detrend button
        self.optionsframe.rowconfigure(6,weight=0)
        self.optionsframe.rowconfigure(7,weight=0)

        self.maingrouplabel = ttk.Label(self.optionsframe, text='main group:').grid(column=0,row=0,padx=(5,5), pady=(5,5),sticky='nws')
        self.maingroupcombobox = ttk.Combobox(self.optionsframe)
        self.maingroupcombobox['values'] = list(self.file.keys())            #['1','2','3']
        self.maingroupcombobox['state'] = 'readonly'
        self.maingroupcombobox.grid(column=0,row=1,padx=(5,5), pady=(2,2),sticky='nesw')

        self.datagrouplabel = ttk.Label(self.optionsframe, text='data category:').grid(column=0,row=2,padx=(5,5), pady=(2,2),sticky='nws')
        self.datagroupcombobox = ttk.Combobox(self.optionsframe)
        self.datagroupcombobox['state'] = 'readonly'
        self.datagroupcombobox.grid(column=0,row=3,padx=(5,5), pady=(2,2),sticky='nesw')

        self.datalabel = ttk.Label(self.optionsframe, text='data:').grid(column=0,row=4,padx=(5,5), pady=(2,2),sticky='nws')
        self.datacombobox = ttk.Combobox(self.optionsframe)
        self.datacombobox['state'] = 'readonly'
        self.datacombobox.grid(column=0,row=5,padx=(5,5), pady=(2,2),sticky='nesw')

        self.detrend_button = ttk.Button(self.optionsframe, text='detrend', command=self.detrend)
        self.detrend_button.grid(column=0, row=6, padx=(5,5), pady=(2,2),sticky='nesw')

        self.diff_button = ttk.Button(self.optionsframe, text='difference', command=self.diff)
        self.diff_button.grid(column=0, row=7, padx=(5,5), pady=(2,2),sticky='nesw')

        ## binding functions
        self.maingroupcombobox.bind('<<ComboboxSelected>>', lambda e: self.update_datagroup_vals())
        self.datagroupcombobox.bind('<<ComboboxSelected>>', lambda e: self.update_data_vals())
        self.datacombobox.bind('<<ComboboxSelected>>', lambda e: self.update_ts())

    def update_datagroup_vals(self):
        self.maingroup.set(self.maingroupcombobox.get())
        self.datagroupcombobox['values'] = list(self.file[self.maingroup.get()].keys())

    def update_data_vals(self):
        self.datagroup.set(self.datagroupcombobox.get())
        self.datacombobox['values'] = list(self.file[self.maingroup.get()][self.datagroup.get()].keys())

    def update_ts(self):
        self.data.set(self.datacombobox.get())
        self.ts = self.file[self.maingroup.get()][self.datagroup.get()][self.data.get()][:]
        self.plot_data = self.ts
        self.update_plot()

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        data = self.plot_data
        # self.ax1.set_title('transaction_data')
        self.ax1.plot(data, color='blue')
        lags, acr_values = self.calc_acr(data, n_lags=40)
        # self.ax2.set_title('autocorrelation')
        self.ax2.stem(lags, acr_values)
        self.ax2.hlines([-1*1.96/np.sqrt(len(data)),1.96/np.sqrt(len(data))],0,len(lags)-1,colors=['r']*2,linestyles=['dashed']*2)
        psd_values, freqs = self.ax3.psd(data, Fs=2)
        self.ax3.clear()
        # self.ax3.set_title('power spectrum density')
        self.ax3.plot(freqs, 10*np.log10(psd_values))

        # self.fig.tight_layout()
        # print(0,len(data))
        # self.ax.set_xlim([0,len(data)])
        # print(min(data), max(data))
        # self.ax.set_ylim([min(data),max(data)])
        self.fig.canvas.draw()

    def init_plot(self):
        # plt.ion()
        fig = plt.Figure(figsize=(16.5,5), dpi=100)
        fig.patch.set_facecolor('#F0F0F0')
        fig.patch.set_alpha(1.0)
        gs = fig.add_gridspec(2, 2)
        fig_ax1 = fig.add_subplot(gs[0,:])
        fig_ax2 = fig.add_subplot(gs[1,0])
        fig_ax3 = fig.add_subplot(gs[1,1])
        fig.tight_layout()
        chart = FigureCanvasTkAgg(fig,self.figureframe)
        chart.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH)
        toolbar=NavigationToolbar2Tk(chart, self.toolbarframe)
        self.fig = fig
        self.ax1 = fig_ax1
        self.ax2 = fig_ax2
        self.ax3 = fig_ax3
        # self.ax1.set_title('transaction_data')
        # self.ax2.set_title('autocorrelation')  
        # self.ax3.set_title('power spectrum density')


    ## calculate autocorrealtion values
    def calc_acr(self,data, n_lags=40):
        lags = np.arange(n_lags+1)
        acr_values = sm.tsa.acf(data)
        return lags, acr_values

    ## calculate power density spectrum
    # def calc_psd(self,data):
    #     F = np.fft.fft(data)
    #     N = 0.5 * len(F)**2
    #     psd_values = (1/N) * F[1:len(F/2)] * np.conj(F[1:len(F/2)])
    #     psd_values = 10 * log10(psd_values)
    #     return psd_values

    ## calculate trend decomposition
    def detrend(self):
        result_add = seasonal_decompose(self.plot_data, model='additive', extrapolate_trend=1, period=30)
        residual = self.plot_data - result_add.trend
        self.plot_data = residual
        self.update_plot()

    def diff(self):
        self.plot_data = np.diff(self.plot_data)
        self.update_plot()

    def reset(self,event):
        self.plot_data = self.ts
        self.update_plot()

    def event_loop(self):
        self.root_window.mainloop()



