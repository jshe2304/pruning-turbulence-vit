import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_analysis(results, analysis_dict, plot_dir):

    font = {'size': 14}
    mpl.rc('font', **font)

    if analysis_dict['rmse']:
        # U
        fig, ax = plt.subplots()

        x = np.arange(1, 1+len(results['rmse_u_median'])) 
        ax.plot(x, results['rmse_u_median'], '-k', label='ML')
        upper = results['rmse_u_uq'] # + results['rmse_u_std']
        lower = results['rmse_u_lq'] # - results['rmse_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_u_per_median'], '--k', label='Persistence')
        upper = results['rmse_u_per_uq'] # + results['rmse_u_per_std']
        lower = results['rmse_u_per_lq'] # - results['rmse_u_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, len(results['rmse_u_median'])])

        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir + '/RMSE_U_' + '.svg')
        # V
        fig, ax = plt.subplots()

        ax.plot(x, results['rmse_v_median'], '-k', label='ML')
        upper = results['rmse_v_uq'] # + results['rmse_v_std']
        lower = results['rmse_v_lq'] # - results['rmse_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_v_per_median'], '--k', label='Persistence')
        upper = results['rmse_v_per_uq'] # + results['rmse_v_per_std']
        lower = results['rmse_v_per_lq'] # - results['rmse_v_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, len(results['rmse_v_median'])])

        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir + '/RMSE_V_' + '.svg')

    if analysis_dict['acc']:
        # U
        fig, ax = plt.subplots()

        x = np.arange(1, 1+len(results['acc_u_median'])) 
        ax.plot(x, results['acc_u_median'], '-k', label='ML')
        upper = results['acc_u_uq'] # + results['acc_u_std']
        lower = results['acc_u_lq'] # - results['acc_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_u_per_median'], '--k', label='Persistence')
        upper = results['acc_u_per_uq'] # + results['acc_u_per_std']
        lower = results['acc_u_per_lq'] # - results['acc_u_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(results['acc_u_median'])])

        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir + '/ACC_U_' + '.svg')
        # V
        fig, ax = plt.subplots()

        ax.plot(x, results['acc_v_median'], '-k', label='ML')
        upper = results['acc_v_uq'] # + results['acc_v_std']
        lower = results['acc_v_lq'] # - results['acc_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_v_per_median'], '--k', label='Persistence')
        upper = results['acc_v_per_uq'] # + results['acc_v_per_std']
        lower = results['acc_v_per_lq'] # - results['acc_v_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, len(results['acc_v_median'])])

        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir + '/ACC_V_' + '.svg')

    if analysis_dict['spectra']:
        fig, ax = plt.subplots()
        x = results['wavenumbers']
        ax.plot(x, results['spectra_tar'][0], '-k', label='Truth')
        for lead in analysis_dict['spectra_leadtimes']:
            spec = results['spectra'][lead]

            label = rf'{lead+1}$\Delta t$' 

            ax.plot(x, spec, label=label)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Wavenumbers')
            ax.set_ylabel('Power')
            ax.set_xlim([0.8, 200])
            ax.set_ylim([10**(-9), 10])
            ax.legend()
            plt.tight_layout()
            fig.savefig(plot_dir + '/Power_Spectra_' + '.svg')
