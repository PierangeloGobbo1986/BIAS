import pandas
import numpy as np
from datetime import datetime
import glob
import os
import os.path
import shutil
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.legend_handler
import matplotlib.collections
import pandas as pd
# import matplotlib.cm as cm
from tkinter import *
from tkinter import scrolledtext
from tkinter import filedialog
# from tkinter.filedialog import asksaveasfilename
# import math
# import warnings
# from matplotlib.colors import colorConverter

# import scipy.interpolate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model
# ----------------------------------------------------------------------------------------------------------------------#
#BASIC FUNCTIONS
def Spacer():
    """Creates a spacer (---) in GUI dialogue window."""
    txt.insert(END, "-"*134)
    txt.insert(END, "\n")
    txt.update()
    txt.see("end")

def Create_Output_Folder(new_folder_name):
    """Creates a new directory with given input folder name."""
    try:
        os.mkdir(new_folder_name) # Make new directory

    # Error handling to avoid stopping code
    except OSError:
        txt.insert(END, "Error: folder {} already present.\n".format(new_folder_name))
        txt.update()
        txt.see("end")

    else:
        txt.insert(END, "Directory {} successfully created.\n".format(new_folder_name))
        txt.update()
        txt.see("end")

def Find_Experiments(path_filename):
    """Divides txt file containing multiple (array or single) measurements into individual txt files for each measurement."""
    # Save txt files in new folder with the name of the txt file
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_name = filename_noext
    folder_path = "{}/{}".format(path, folder_name)
    Create_Output_Folder(folder_path) #This is for the experiment


    inputfile = open(path_filename, 'r') #, encoding='gbk'
    txt.insert(END, "Working directory:\n{}\n\n".format(path))
    txt.insert(END, "File opened: {}\n\n".format(filename))
    txt.update()
    txt.see("end")
    Spacer()

    fileno = -1
    outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w") #This is the new f-string method of Python 3.6
    txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
    txt.update()
    txt.see("end")
    for line in inputfile:
        if not line.strip():
            fileno += 1
            outfile.close()
            outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w")
            txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
            txt.update()
            txt.see("end")
        else:
            outfile.write(line)
    outfile.close()
    inputfile.close()

    #Delete data_-1.txt file which is useless
    os.remove(f"{folder_path}/{filename_noext}_-1.txt")
    txt.insert(END, f"File {folder_path}/{filename_noext}_-1.txt deleted.\n")
    txt.update()
    txt.see("end")


    txt.insert(END, "Individual measurements successfully detected.\n")
    txt.insert(END, "Detected {} measurements.\n\n".format(fileno+1))
    txt.update()
    txt.see("end")
    return path, filename, filename_noext, extension, folder_path

def Divide_Array_Data(path_filename, folder_path_Array_Measurements):
    """Divides a txt file containing an array of indentation measurements into individual csv files for each indentation."""
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    # folder_path_Array_Measurements = path + f"/{filename_noext}_Measurements"
    # Create_Output_Folder(folder_path_Array_Measurements)

    #Convert data to df
    df = pd.read_table(path_filename, low_memory=False, encoding = "unicode_escape", on_bad_lines='skip', delim_whitespace=True, names=("Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
            "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
            "Voltage A [V]", "Voltage B [V]", "Temperature [oC]", "Amplitude force [uN]", "Amplitude pos [um]",
            "Stiffness [N/m]", "Phase shift [°]", "Phase excitation [°]", "Force A raw [uN]", "Displacement raw [um]",
            "HMax [um]", "HMax raw [um]", "Real Force [uN]", "Real Stiffness [N/m]", "Contact Depth [um]",
            "Area [um^2]", "Hardness [MPa]", "Reduced Mod. [MPa]"))
    txt.insert(END, f"DF: {df}\n")
    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
    #df = df.replace('//', np.nan, regex=True).replace('=', np.nan, regex=True)
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float

    #txt.insert(END, f"DF: {df}\n")

    num_mes = int(df["Index [#]"].max() + 1)
    txt.insert(END, f"Number of individual measurements found: {str(num_mes)}\n")
    txt.update()
    txt.see("end")

    grouped = df.groupby(["Index [#]"])  # Group full database by measurement#

    for num in range(num_mes):
        #txt.insert(END, f"{num + 1}) Plotting file {filename_noext}_{str(num + 1)}...\n")
        #txt.update()
        #txt.see("end")

        group = pd.DataFrame(grouped.get_group(num))  # Get a single experiement
        #phase = []
        #piezoZ = group["Piezo Z [um]"].to_list()  # Use PiezoZ for determining phase number
        # phase_num = 1
        # for ind, val in enumerate(piezoZ):  # enumerate loops over list (val) and have an automatic counter (ind).
        #     try:
        #         if piezoZ[ind + 1] - piezoZ[ind] >= 0:
        #             phase.append(phase_num)
        #             phase_num = 1
        #
        #         else:
        #             phase.append(phase_num)
        #             phase_num = 0
        #     except:
        #         phase.append(phase_num)  # Fixes the last loop where there is not a num+1 to subtract
        #
        # group["Exp Phase [#]"] = phase  # converts Displacement column into a list and applies function Cycle
        group.to_csv(f"{folder_path_Array_Measurements}/{filename_noext}_{str(num + 1)}.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

def DetermineIndentationDepth(df, path, filename):
    global Time_CP_Sel, Force_CP_Sel, fit_result, cp, a, b, exponent, c, d, e
    filename_noext, extension = os.path.splitext(filename)
    folder_path_Indentation_Depth = path + f"/Indentation_Depth_Analysis"

    Time_col = df.columns.get_loc("Time [s]")  # Get column number
    Force_col = df.columns.get_loc("Force A [uN]")  # Get column number
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")  # Get column number
    Phase_col = df.columns.get_loc("Phase [#]")

    Time = np.array(df.iloc[:, Time_col]) #define array of time data
    Force = np.array(df.iloc[:, Force_col])  #define array of force data
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])  # define array of Piezo Z data
    Phase = np.array(df.iloc[:, Phase_col])

    # DETERMINE MEASUREMENT PHASES, AND CALCULATE DISPLACEMENT DURING MEASUREMENT
    F_max_index = np.argmax(Force) # index of maximum force

    # Optional index adjustment for F_max_index:
    # Occasionally piezoscanner will overshoot then return slightly - the sharp decrease in F from this process should be excluded from analysis
    #meanPiezoZ, sdPiezoZ = np.mean(PiezoZ[F_max_index:MoveBack_Index]), np.std(PiezoZ[F_max_index:MoveBack_Index]) # Calculate mean + SD of piezo Z channel during constant indentation
    #PiezoZmax = meanPiezoZ + 100 * sdPiezoZ # Set a threshold piezo Z value as (mean + SD * multiple). 100 as default has no effect on data.
    #Index_Adjustment = np.array([np.where(PiezoZ[F_max_index:MoveBack_Index] < PiezoZmax)]).min(initial=np._NoValue) # First index of PiezoZ < PiezoZmax (i.e. where PiezoZ is stable)
    #PiezoZ_Max_index = F_max_index + Index_Adjustment # Adjust F_max_index to where PiezoZ is stable

    Displacement_Max_um = PiezoZ[F_max_index] # Define maximum displacement

    txt.insert(END, "Maximum F index: {}\n\n".format(str(F_max_index)))
    txt.insert(END, "Maximum displacement: {} um\n\n".format(str(round(Displacement_Max_um, 3))))
    txt.update()
    txt.see("end")

    # CALCULATE CONTACT POINT:

    # DETERMINE SMOOTHING WINDOW BASED ON NUMBER OF DATA POINTS DURING LOADING
    SG_windowlength = round(len(Force[0:F_max_index])/15)
    if SG_windowlength % 2 == 0: # Ensures window length is odd (required for Savitsky-Golay filter)
        SG_windowlength = SG_windowlength + 1
    if SG_windowlength < 3: # Ensures window length is longer than polynomial order
        SG_windowlength = 3
    txt.insert(END, "SG window length: {}\n\n".format(str(SG_windowlength)))

    # SMOOTH DATA AND ESTIMATE CP AS MAXIMUM IN 2ND DERIVATIVE
    Force_Interp = savgol_filter((Force[0:F_max_index]), SG_windowlength, 2) # Interpolate data (Savitsky Golay)
    Est_SecondDeriv_Force = savgol_filter((Force[0:F_max_index]), SG_windowlength, 2, deriv=2) # Interpolate and calculate 2nd derivative of data
    Est_CP_index = np.argmax(Est_SecondDeriv_Force[0:F_max_index]) # Estimates index of contact point as maximum in 2nd derivative of smoothed data

    # SELECT REGION AROUND ESTIMATED CP FOR MORE ACCURATE FITTING
    # nb. estimated CP tends to be later than actual CP so window is skewed towards before estimated CP.
    CP_window = round((len(Force[0:F_max_index]))/8)
    CP_Selection_Min = Est_CP_index - 3*CP_window # If CP determination is inconsistent, selection min/max can be varied.
    CP_Selection_Max = Est_CP_index + CP_window
    if CP_Selection_Max > F_max_index: # Bound CP window between start of data and F_max_index
        CP_Selection_Max = F_max_index
    if CP_Selection_Min < 1:
        CP_Selection_Min = 1

    # PIECEWISE FITTING TO GIVE ACCURATE CONTACT POINT:
    # Selected data around the estimated CP are fit to linear and exponential regions with the
    # switchpoint between regions a fitting parameter which should be equivalent to the CP.
    try:
        Time_CP_Sel = Time[CP_Selection_Min:CP_Selection_Max]-Time[0]
        Force_CP_Sel = Force[CP_Selection_Min:CP_Selection_Max]-Force[0]

        from symfit import parameters, variables, Fit, Piecewise, Eq # exp,
        # from symfit.core.minimizers import DifferentialEvolution, NelderMead, BFGS, ScipyConstrainedMinimize # Alternative minimisers for symfit module

        # Define fitting parameters and x/y input (time/force)
        t, y = variables('t, y')
        a, b, c, d, e, cp = parameters('a, b, c, d, e, cp')

        # Help the fit by bounding the switchpoint between the models and using the estimated CP as starting switchpoint
        cp.min = np.amin(Time_CP_Sel)
        cp.max = np.amax(Time_CP_Sel)
        exponent = 1.5 # Vary exponent if needed (1.5 by default)
        cp.value = np.mean(Time_CP_Sel)

        # Make a piecewise model. Linear when t < cp, exponential when t >= cp.
        # Exponent can be varied but 1.5 is chosen based on the Hertz model (assuming time scales linearly with displacement)
        y1 = a * t + b
        y2 = c * t ** exponent + d * t + e
        model = {y: Piecewise((y1, t < cp), (y2, t >= cp))}

        # As a constraint, we demand equality between the two models at the point cp
        # to do this, we substitute t -> cp and demand equality using `Eq`
        constraints = [Eq(y1.subs({t: cp}), y2.subs({t: cp}))]

        # Alternative constraints: continuous derivative at point cp
        # constraints = [Eq(y1.diff(t).subs({t: cp}), y2.diff(t).subs({t: cp})), Eq(y1.subs({t: cp}), y2.subs({t: cp}))]

        # FIT DATA TO PIECEWISE MODEL WITH GIVEN CONSTRAINTS
        # The determined value of 'cp' is within 'fit_result.
        # Fitting methods:

        # 1. Using standard minimizer (least-squares) is fastest but may find local minima influenced by initial parameters:
        fit = Fit(model, t=Time_CP_Sel, y=Force_CP_Sel, constraints=constraints)

        # 2. BasinHopping algorithm is more computationally expensive but should do a better job of finding global minimum:
        # from symfit.core.minimizers import BasinHopping
        # fit = Fit(model, t=Time_CP_Sel, y=Force_CP_Sel, constraints=constraints, minimizer=BasinHopping)

        fit_result = fit.execute()

        # Determine the index (from the full df) of the CP, as the value before t > t(cp)
        Fitted_CP_index = (next(x for x, val in enumerate(Time-Time[0])
                             if val > (fit_result.value(cp)))) - 1

        fitted_cp = 1

    except(ValueError): #
        Fitted_CP_index = Est_CP_index
        fitted_cp = 0
        txt.insert(END, "Contact point fitting failed. Using estimated contact point from maximum second derivative.")
        Spacer()

    # TO CHECK CP FITTING
    # txt.insert(END, "CP fit index: {} [s]\n\n".format(str((Fitted_CP_index))))
    # txt.insert(END, "CP fit result: {} [s]\n\n".format(str((fit_result))))
    # txt.insert(END, "CP time fitted: {} [s]\n\n".format(str((fit_result.value(cp)))))

    # Determine parameters at CP
    fitted_cp_time = Time[Fitted_CP_index]
    F_Min_uN = Force[Fitted_CP_index]
    Displacement_Min_um = PiezoZ[Fitted_CP_index]
    Max_Indentation_Depth_um = float(Displacement_Max_um - Displacement_Min_um) # determines maximum indentation depth
    txt.insert(END, "Maximum indentation depth: {} um\n\n".format(str(np.round(Max_Indentation_Depth_um, 3))))
    txt.update()
    txt.see("end")

    Ramp_Time = float(Time[F_max_index]-Time[Fitted_CP_index])

    #Add calculated values to dataframe and save as csv
    df['Filtered time [s]'] = pd.Series(Time[:F_max_index])
    df['Filtered force [uN]'] = pd.Series(Force[:F_max_index])
    df['SG Force [uN]'] = pd.Series(Force_Interp)
    df['Est Force 2nd derivative'] = pd.Series(Est_SecondDeriv_Force)
    # df['Force 2nd derivative'] = pd.Series(SecondDeriv_Force)
    df['Filtered displacement [um]'] = pd.Series(PiezoZ[:F_max_index])
    df.to_csv(f"{folder_path_Indentation_Depth}/{filename_noext}_IndentationDepthCalculation.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    # PLOT INDENTATION DEPTH DETERMINATION
    # 1. Estimated CP based on 2nd derivative
    x, y2, y3 = (Time[:F_max_index]-Time[0]), Est_SecondDeriv_Force, Force[:F_max_index]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Force derivative', fontsize=18)
    #lns1 = ax1.plot(x, y1, color='black', label='Est 1st Derivative')
    lns2 = ax1.plot(x, y2, color='blue', label='Est 2nd Derivative')
    ax1.tick_params(axis = 'y', labelcolor='black')
    ax1.set_xscale('linear')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns3 = ax2.plot(x, y3, color='red', label='Force_SG')
    lns4 = ax2.axvline(x=(Time[Est_CP_index]-Time[0]), color='k', linestyle='--', label='Contact Point')
    ax2.tick_params(axis = 'y', labelcolor='red')

    lns = lns2+lns3 #To handle combined figure legend
    labs = [l.get_label() for l in lns] #To handle combined figure legend
    ax1.legend(lns, labs, loc=0, fontsize=12) #To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_CP_approx"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_CP_approx.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # 2. Piecewise CP fitting (if successful)
    if fitted_cp == 1:
        xi, yi = Time_CP_Sel, Force_CP_Sel
        xii = np.linspace(np.amin(Time_CP_Sel), fit_result.value(cp), num=200)
        yii = xii * fit_result.value(a) + fit_result.value(b)
        xiii = np.linspace(fit_result.value(cp), np.amax(Time_CP_Sel), num=200)
        yiii = xiii ** exponent * fit_result.value(c) + fit_result.value(d) * xiii + fit_result.value(e) # Exponent to match fitting equation
        fig, ax1 = plt.subplots()

        txt.insert(END, "CP fit result params: {}\n\n".format(str((fit_result.params))))

        ax1.set_xlabel('Time (s)', fontsize=18)
        ax1.set_ylabel('Force (uN)', fontsize=18)
        lns1 = ax1.plot(xi, yi, 'ko', label='Raw data') #color='black'
        #lns2 = ax1.plot(xii, model(t=xii, **fit_result.params).y, color='blue', linestyle='--', label='Piecewise Fit')
        lns2 = ax1.plot(xii, yii, color='blue', linestyle='-', label='Piecewise Fit - Linear')
        lns3 = ax1.plot(xiii, yiii, color='red', linestyle='-', label='Piecewise Fit - Exp')
        ax1.tick_params(axis = 'y', labelcolor='black')
        ax1.set_xscale('linear')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        lns4 = ax1.axvline(x=fit_result.value(cp), color='g', linestyle='--', label='Contact Point (Fitted)')

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
        # lns3 = ax2.plot(x1, y2, color='red', label='Force')
        # lns4 = ax2.plot(x2, y4, 'b--', label='Force_Interpolated')
        # lns5 = ax2.axvline(x=(CP_time-Time[0]), color='k', linestyle='--', label='Contact Point')
        # ax2.tick_params(axis = 'y', labelcolor='red')

        # lns = lns1+lns3 #To handle combined figure legend
        # labs = [l.get_label() for l in lns] #To handle combined figure legend
        # ax1.legend(lns, labs, loc=0, fontsize=12) #To handle combined figure legend
        plt.yticks(fontsize=12)
        plt.title("{}".format(f"{filename_noext}_CP_precise"), fontsize=20)
        fig.tight_layout()
        plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_CP_precise.png", bbox_inches='tight', dpi=300)
        plt.close('all') #To close all figures and save memory - max # of figures before warning: 20
    elif fitted_cp == 0:
        pass

    # 3. Indentation depth analysis
    x, y1, y2 = Time[:round(F_max_index+(F_max_index/10))]-Time[0], PiezoZ[:round(F_max_index+(F_max_index/10))], Force[:round(F_max_index+(F_max_index/10))]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Piezo Z (um)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Piezo Z')
    ax1.tick_params(axis = 'y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns2 = ax2.plot(x, y2, color='red', label='Force')
    lns3 = ax2.axvline(x=Time[Est_CP_index]-Time[0], color='k', linestyle='--', label='Contact Point (approx)')
    lns4 = ax2.axvline(x=Time[F_max_index]-Time[0], color='b', linestyle='--', label='Maximum Force')
    lns5 = ax2.axvline(x=fitted_cp_time-Time[0], color='g', linestyle='--', label='Contact Point (Fitted)')
    ax2.tick_params(axis = 'y', labelcolor='red')

    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.6))
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 0.5))

    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Indentation Depth Analysis"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_Indentation_Depth_Analysis.pdf", bbox_inches='tight')
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    return Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force, Est_CP_index, Fitted_CP_index

def CalculateBerkovichResults_contact (path_filename, v_sample, C0, C1, C2, C3, C4, C5, Beyeler_depth, Beta): #, threshold_constant. This code evaluate a new precise contact point using the function above

    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_Berkovich = path + f"/Berkovich Results"

    # Convert raw txt data to df
    try:
        df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                           delim_whitespace=True, names=(
                "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
            "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
            "Voltage A [V]", "Voltage B [V]", "Temperature [oC]", "Amplitude force [uN]", "Amplitude pos [um]",
            "Stiffness [N/m]", "Phase shift [°]", "Phase excitation [°]", "Force A raw [uN]", "Displacement raw [um]",
            "HMax [um]", "HMax raw [um]", "Real Force [uN]", "Real Stiffness [N/m]", "Contact Depth [um]",
            "Area [um^2]", "Hardness [MPa]", "Reduced Mod. [MPa]"))
        df = df[~df["Index [#]"].isin(
            ['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
        df = df.dropna(how='all')  # to drop if all values in the row are nan
        df = df.astype(float)  # Change data from object to float
    # Convert previously processed csv file to df
    except:
        df = pd.read_csv(path_filename)
        pass

    Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force, Est_CP_index, Fitted_CP_index = DetermineIndentationDepth(df, path, filename)

    # Get other columns required for data analysis
    Phase_col = df.columns.get_loc("Phase [#]")
    Force_col = df.columns.get_loc("Force A [uN]")
    Hardness_col = df.columns.get_loc("Hardness [MPa]")
    Red_Modulus_col = df.columns.get_loc("Reduced Mod. [MPa]")
    Real_force_col = df.columns.get_loc("Real Force [uN]")
    Real_stiffness_col = df.columns.get_loc("Real Stiffness [N/m]")
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")
    Stiffness_col = df.columns.get_loc("Stiffness [N/m]")
    Contact_depth_first_col=df.columns.get_loc("Contact Depth [um]")
    Displacement_col = df.columns.get_loc("Displacement [um]")
    # X_col = df.columns.get_loc("Pos X [um]")
    # Y_col = df.columns.get_loc("Pos Y [um]")

    Phase = np.array(df.iloc[:, Phase_col])  # define array of time data
    Force = np.array(df.iloc[:, Force_col])
    Hardness = np.array(df.iloc[:, Hardness_col])
    Red_Modulus = np.array(df.iloc[:, Red_Modulus_col])
    Real_force = np.array(df.iloc[:, Real_force_col])
    Real_stiffness = np.array(df.iloc[:, Real_stiffness_col])
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])
    Stiffness = np.array(df.iloc[:, Stiffness_col])
    Contact_depth_first = np.array(df.iloc[:, Contact_depth_first_col])
    Displacement = np.array(df.iloc[:, Displacement_col])
    # X = np.array(df.iloc[:, X_col])
    # Y = np.array(df.iloc[:, Y_col])


    size = round(len(Displacement[Fitted_CP_index:]) * 0.3)
    index_contact_new = Fitted_CP_index + size # The data from index_contact_new till the end correspond to the last 70% of data points where Hardness and Reduced Modulus are stable. It excludes the first 30% of data immediately after contact point.
    contact_point = Displacement[Fitted_CP_index]

    contact_depth = (Displacement[index_contact_new:] - contact_point) - ( 0.75 * (Real_force[index_contact_new:] / Real_stiffness[index_contact_new:]))

    Force_selected = Force[index_contact_new:]
    Stiffness_selected = Stiffness[index_contact_new:]

    Area_function = (C0 * contact_depth ** 2) + (C1 * 1 ** 2) * Beyeler_depth * contact_depth * np.exp((-1) * contact_depth / Beyeler_depth) + (C2 * 2 ** 2) * Beyeler_depth * contact_depth * np.exp((-2) * contact_depth / Beyeler_depth) + (C3 * 3 ** 2) * Beyeler_depth * contact_depth * np.exp((-3) * contact_depth / Beyeler_depth) + ( C4 * 4 ** 2) * Beyeler_depth* contact_depth * np.exp((-4) * contact_depth / Beyeler_depth) + (C5 * 5 ** 2) * Beyeler_depth * contact_depth * np.exp((-5) * contact_depth / Beyeler_depth)


    Hardness_new = Force_selected/Area_function
    Red_Modulus_new= (Stiffness_selected/(Beta*2))*((np.pi/Area_function)**(0.5))

    df['Force selected [uN]'] = pd.Series(Force_selected)
    df['Contact depth new [um]'] = pd.Series(contact_depth)
    df['Area New [um^2]'] = pd.Series(Area_function)
    df['Hardness New [MPa]'] = pd.Series(Hardness_new)
    df['Red Mod New [MPa]'] = pd.Series(Red_Modulus_new)
    Hardness_mean = df.loc[:, 'Hardness New [MPa]'].mean()
    Red_Modulus_mean = df.loc[:, 'Red Mod New [MPa]'].mean()
    Hardness_Err = df.loc[:, 'Hardness New [MPa]'].std()
    Red_Mod_Err = df.loc[:, 'Red Mod New [MPa]'].std()
    v_tip=0.07
    Young_Mod = (1-(v_sample**2))/(((-(1-(v_tip**2)))/(1140*1000))+(1/Red_Modulus_new))
    df['Young Mod [MPa]'] = pd.Series(Young_Mod)
    Young_Mod_mean = df.loc[:, 'Young Mod [MPa]'].mean()
    Young_Mod_Err = df.loc[:, 'Young Mod [MPa]'].std()
    X_mean = df.loc[:, 'Pos X [um]'].mean()
    Y_mean = df.loc[:, 'Pos Y [um]'].mean()

    df.to_csv(f"{folder_path_Berkovich}/{filename_noext}_Berkovich.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index


    # GET HEIGHT DATA
    PosZ_CP = np.float64(df["Pos Z [um]"].median())
    PiezoZ_CP = Displacement_Min_um


    return path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point

def CalculateBerkovichResults_NoContact (path_filename, v_sample): #This code assumes as contact point the one found by the FemtoTool software

    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_Berkovich = path + f"/Berkovich Results"

    # Convert raw txt data to df
    try:
        df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                           delim_whitespace=True, names=(
                "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
            "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
            "Voltage A [V]", "Voltage B [V]", "Temperature [oC]", "Amplitude force [uN]", "Amplitude pos [um]",
            "Stiffness [N/m]", "Phase shift [°]", "Phase excitation [°]", "Force A raw [uN]", "Displacement raw [um]",
            "HMax [um]", "HMax raw [um]", "Real Force [uN]", "Real Stiffness [N/m]", "Contact Depth [um]",
            "Area [um^2]", "Hardness [MPa]", "Reduced Mod. [MPa]"))
        df = df[~df["Index [#]"].isin(
            ['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
        df = df.dropna(how='all')  # to drop if all values in the row are nan
        df = df.astype(float)  # Change data from object to float
    # Convert previously processed csv file to df
    except:
        df = pd.read_csv(path_filename)
        pass


    # Get other columns required for data analysis
    Phase_col = df.columns.get_loc("Phase [#]")
    Force_col = df.columns.get_loc("Force A [uN]")
    Hardness_col = df.columns.get_loc("Hardness [MPa]")
    Red_Modulus_col = df.columns.get_loc("Reduced Mod. [MPa]")
    Real_force_col = df.columns.get_loc("Real Force [uN]")
    Real_stiffness_col = df.columns.get_loc("Real Stiffness [N/m]")
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")
    Stiffness_col = df.columns.get_loc("Stiffness [N/m]")
    Area_col = df.columns.get_loc("Area [um^2]")
    Contact_depth_first_col = df.columns.get_loc("Contact Depth [um]")
    Displacement_col = df.columns.get_loc("Displacement [um]")

    # define array of specific data
    Phase = np.array(df.iloc[:, Phase_col])
    Force = np.array(df.iloc[:, Force_col])
    Hardness = np.array(df.iloc[:, Hardness_col])
    Red_Modulus = np.array(df.iloc[:, Red_Modulus_col])
    Real_force = np.array(df.iloc[:, Real_force_col])
    Real_stiffness = np.array(df.iloc[:, Real_stiffness_col])
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])
    Stiffness = np.array(df.iloc[:, Stiffness_col])
    Area = np.array(df.iloc[:, Area_col])
    Contact_depth_first = np.array(df.iloc[:, Contact_depth_first_col])
    Displacement = np.array(df.iloc[:, Displacement_col])

    # index_contact = df[df['Phase [#]'] == 2].index.values
    index_contact = np.array([np.where(Phase >= 2)]).min()
    #index_max_force = np.argmax(Force)

    size = round(len(Phase[index_contact:]) * 0.3)
    index_contact_new = index_contact + size # The data from index_contact_new till the end correspond to the last 70% of data points where Hardness and Reduced Modulus are stable. It excludes the first 30% of data immediately after contact point.

    contact_point=Displacement[index_contact]

    Hardness_selected = Hardness[index_contact_new:]
    Red_Modulus_selected = Red_Modulus[index_contact_new:]
    df['Area selected [um^2]'] = pd.Series(Area[index_contact_new:])
    df['Force selected [uN]'] = pd.Series(Force[index_contact_new:])
    df['Sel Hardness [MPa]'] = pd.Series(Hardness_selected)
    df['Sel Red Modulus [MPa]'] = pd.Series(Red_Modulus_selected)
    Hardness_mean=df.loc[:, 'Sel Hardness [MPa]'].mean()
    #Hardness_mean = np.mean(Hardness_selected[index_contact_new:])
    Red_Modulus_mean=df.loc[:, 'Sel Red Modulus [MPa]'].mean()
    Hardness_Err = df.loc[:, 'Sel Hardness [MPa]'].std()
    Red_Mod_Err = df.loc[:, 'Sel Red Modulus [MPa]'].std()
    v_tip=0.07
    Young_Mod = (1-(v_sample**2))/(((-(1-(v_tip**2)))/(1140*1000))+(1/Red_Modulus_selected))
    df['Sel Young Mod [MPa]'] = pd.Series(Young_Mod)
    Young_Mod_mean = df.loc[:, 'Sel Young Mod [MPa]'].mean()
    Young_Mod_Err = df.loc[:, 'Sel Young Mod [MPa]'].std()
    X_mean = df.loc[:, 'Pos X [um]'].mean()
    Y_mean = df.loc[:, 'Pos Y [um]'].mean()

    df.to_csv(f"{folder_path_Berkovich}/{filename_noext}_Berkovich.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    # GET HEIGHT DATA
    PosZ_CP = np.float64(df["Pos Z [um]"].median())
    PiezoZ_CP=PiezoZ[index_contact]

    return path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point
# ----------------------------------------------------------------------------------------------------------------------#
#GUI
# ----------------------------------------------------------------------------------------------------------------------#

#Run Software
root  = Tk()
root.wm_title("Gobbo Group – B.I.A.S.") #B.I.A.S. = Berkovich Indentation Analysis System
root.geometry("1350x650")
#root.configure(bg="white")


#SET VARIABLES TO BE USED IN OTHER BUTTONS
#NOTE: Tkinter button cannot return a variable. To modify these variables inside the button's function and make them available outside
# the button's function you need to declare them outside the function and then declare them as global inside the function.
path_filename = ""
path = ""
filename = ""
filename_noext = ""
extension = ""
folder_path = ""
path_foldername = ""

year_now = datetime.now().strftime('%Y')
month_now = datetime.now().strftime("%B")  # returns the full name of the month as a string


# SOFTWARE FUNCTIONS
def File_Extractor_b1():
    """Button to carry out extraction of data.txt files from within a folder of indentation experiments."""
    dir = filedialog.askdirectory(title="Select a folder")
    # to handle Cancel button
    if dir == "":
        return

    txt.insert(END, f"Working directory: {dir}\n\n")
    txt.update()
    txt.see("end")

    #Find all folders within directory
    folder_list = glob.glob(f"{dir}/*/")

    counter_found = 0
    for folder_path_name in folder_list:
        elements = folder_path_name.split("\\")
        folder_name = elements[len(elements)-2]
        txt.insert(END, f"Extracting from: {folder_name}\n")
        txt.update()
        txt.see("end")

        found = 0
        for root, dirs, files in os.walk(folder_path_name, topdown=False):
            for name in files:
                # print(os.path.join(root, name))
                if name == "data.txt":
                    if os.path.isfile(f"{dir}/{folder_name}.txt") == True:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}_{counter_found+1}.txt")
                    else:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}.txt")
                    txt.insert(END, f"File found: {name}.\n")
                    txt.update()
                    txt.see("end")
                    counter_found = counter_found+1
                    found = 1

        if found == 0:
            txt.insert(END, f"Error: No file data.txt found.\n")
            txt.update()
            txt.see("end")

        txt.insert(END, f"Moving {folder_name} to backup folder 'Original data'\n")
        txt.insert(END, f"Operation successful!\n\n")
        txt.update()
        txt.see("end")
        shutil.copytree(f"{folder_path_name[:-1]}", f"{dir}/Original data/{folder_name}")
        shutil.rmtree(f"{folder_path_name[:-1]}")

    txt.insert(END, f"Found {counter_found} data.txt files in {len(folder_list)} experiment folders.\n\n")
    txt.update()
    txt.see("end")
    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def BerkovichAnalysis_Individual_b4():
    """Button to carry out analysis of an array of force relaxation measurements.
    Inputs: probe diameter, Poisson's Ratio, model.
    Options: correct force drift, apply ramp correction faction (RCF).
    Outputs: csv files with output data from model, individual plots of contact point determination and model fitting."""

    # Set variables required
    global path_filename, path, filename, filename_noext, extension

    model = clicked.get()
    #R = (float(e1.get())) / 2
    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes = [("TXT files", "*.txt")])
    v_sample = float(e10.get())
    C0 = float(e2.get())
    C1 = float(e3.get())
    C2 = float(e4.get())
    C3 = float(e5.get())
    C4 = float(e6.get())
    C5 = float(e7.get())
    Beyeler_depth = float(e8.get())
    Beta = float(e9.get())

    if path_filenames == "":
        return

    txt.insert(END, f"Analysing selected txt files...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    if model == "Find Contact":
        Spacer()
        Spacer()
        txt.insert(END, "Processing data finding new contact...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)

            Create_Output_Folder(path + f"/Berkovich Results")
            break #Prevents repeated attempts to create same folder

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)

            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
            break  # Prevents repeated attempts to create same folder

        Labels_list = ['File name', 'X mean [um]', 'Y mean [um]', 'Hardness mean [MPa]', 'Hardness Err [MPa]', 'Red Modulus mean [MPa]', 'Red Mod Err [MPa]', 'Young Mod mean [MPa]', 'Young Mod Err [MPa]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'Contact point [um]'] #List of column headings for results csv file
        Results_list = [] #List of determined parameters for results csv file

        for path_filename in path_filenames:
            path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point = CalculateBerkovichResults_contact(path_filename, v_sample, C0, C1, C2, C3, C4, C5, Beyeler_depth, Beta)
            Results_list.append([filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point])

        with open("{}/Fitting Results Summary_Berkovich.csv".format(path), 'w') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_Berkovich.csv\n".format(path))

    if model == "No Contact":
        Spacer()
        Spacer()
        txt.insert(END, "Processing data finding new contact...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)

            Create_Output_Folder(path + f"/Berkovich Results")
            break  # Prevents repeated attempts to create same folder

        # for path_filename in path_filenames:
        #     path, filename = os.path.split(path_filename)
        #
        #     Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
        #     break  # Prevents repeated attempts to create same folder

        Labels_list = ['File name', 'X mean [um]', 'Y mean [um]', 'Hardness mean [MPa]', 'Hardness Err [MPa]', 'Red Modulus mean [MPa]',
                       'Red Mod Err [MPa]', 'Young Mod mean [MPa]', 'Young Mod Err [MPa]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'Contact point [um]']  # List of column headings for results csv file
        Results_list = []  # List of determined parameters for results csv file

        for path_filename in path_filenames:
            path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point = CalculateBerkovichResults_NoContact(
                path_filename, v_sample)  # if C0, C1,.... are used they need to be placed here
            Results_list.append(
                [filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean,
                 Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point])

        with open("{}/Fitting Results Summary_Berkovich.csv".format(path), 'w') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_Berkovich.csv\n".format(path))

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def BerkovichAnalysis_Array_b5():
    """Button to carry out analysis of an array of force relaxation measurements.
    Inputs: probe diameter, Poisson's Ratio, model.
    Options: correct force drift, apply ramp correction faction (RCF), show measurement locations.
    Outputs: 2D maps of viscoelastic parameters, csv files with output data from model,
    individual plots of contact point determination and model fitting."""

    # Set global variables to use in other functions
    global path_filename, path, filename, filename_noext, extension, csv_filename_noext, PosX_new

    # Define parameters required for analysis
    model = clicked.get()
    #R = (float(e1.get())) / 2

    #show_locations = var3.get()
    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes = [("CSV files", "*.csv")])
    v_sample = float(e10.get())
    C0 = float(e2.get())
    C1 = float(e3.get())
    C2 = float(e4.get())
    C3 = float(e5.get())
    C4 = float(e6.get())
    C5 = float(e7.get())
    Beyeler_depth = float(e8.get())
    Beta = float(e9.get())

    if path_filenames == "":
        return

    txt.insert(END, f"Converting *.txt files to *.csv files and analysing indentation depth...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    if model == "Find Contact":
        Spacer()
        Spacer()
        txt.insert(END, "Processing data finding new contact...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)

            Create_Output_Folder(path + f"/Berkovich Results")
            #Create_Output_Folder(path + f"/Summary")
            break

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)

            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
            break  # Prevents repeated attempts to create same folder

            # Generate lists, labels and dataframes required for reporting data
        Labels_list = ['File name', 'X mean [um]', 'Y mean [um]', 'Hardness mean [MPa]', 'Hardness Err [MPa]', 'Red Modulus mean [MPa]', 'Red Mod Err [MPa]', 'Young Mod mean [MPa]', 'Young Mod Err [MPa]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'Contact point [um]'] #List of column headings for results csv file
        Results_list = [] #List of determined parameters for results csv file


        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {}...\n".format(filename))
            Spacer()

            try:
                path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point = CalculateBerkovichResults_contact (path_filename, v_sample, C0, C1, C2, C3, C4, C5, Beyeler_depth, Beta)
            except:
                df = pd.read_csv(path_filename)
                pass

            Results_list.append([filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point])

        with open("{}/Fitting Results Summary_Berkovich.csv".format(path), 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(Labels_list)
                write.writerows(Results_list)

    if model == "No Contact":

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {} with Generalised Maxwell Model...\n".format(filename))
            Spacer()
            Create_Output_Folder(path + f"/Berkovich Results")
            # Create_Output_Folder(path + f"/Summary")
            break

        # for path_filename in path_filenames:
        #     path, filename = os.path.split(path_filename)
        #
        #     Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
        #     break  # Prevents repeated attempts to create same folder

            # Generate lists, labels and dataframes required for reporting data
        Labels_list = ['File name', 'X mean [um]', 'Y mean [um]', 'Hardness mean [MPa]', 'Hardness Err [MPa]', 'Red Modulus mean [MPa]',
                       'Red Mod Err [MPa]', 'Young Mod mean [MPa]', 'Young Mod Err [MPa]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'Contact point [um]']  # List of column headings for results csv file
        Results_list = []  # List of determined parameters for results csv file

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {}...\n".format(filename))
            Spacer()

            try:
                path, filename_noext, X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean, Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point = CalculateBerkovichResults_NoContact(
                    path_filename, v_sample)  # if C0, C1,.... are used they need to be placed here
            except:
                df = pd.read_csv(path_filename)
                pass

            Results_list.append(
                [filename_noext,  X_mean, Y_mean, Hardness_mean, Hardness_Err, Red_Modulus_mean, Red_Mod_Err, Young_Mod_mean,
                 Young_Mod_Err, PosZ_CP, PiezoZ_CP, contact_point])

        with open("{}/Fitting Results Summary_Berkovich.csv".format(path), 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def Divide_Array_Data_b3():
    """Button to divide txt file(s) containing an array of indentation measurements
     into individual csv files for each indentation."""
    # Set global variables
    global path_filename, path, filename, filename_noext, extension, folder_path_Array_Measurements

    # OPEN FILE
    path_filenames = filedialog.askopenfilenames(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))
    if path_filenames == "":
        return

    # Create a single output folder to store all individual measurements within
    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        filename_noext, extension = os.path.splitext(filename)
        folder_path_Array_Measurements = path + f"/{filename_noext}_Measurements"
        Create_Output_Folder(folder_path_Array_Measurements)
        break

    Spacer()
    txt.insert(END, "Dividing array files into individual *.txt files for each measurement. Number of array files to process: {}\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")
    Spacer()

    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        Spacer()
        txt.insert(END, "Dividing array file {} into individual *.txt files for each measurement...\n".format(filename))
        txt.update()
        txt.see("end")

        Divide_Array_Data(path_filename, folder_path_Array_Measurements)

    Spacer()
    Spacer()
    txt.insert(END, "Individual measurement files saved in directory: {}\n".format(folder_path_Array_Measurements))
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def Gen_Single_Exp_b2():
    """Button to divide txt file containing multiple (array or single) measurements
    into individual txt files for each measurement."""
    # Set global variables
    global path_filename, path, filename, filename_noext, extension, folder_path

    # Ask to open file and get filename and path
    path_filename = filedialog.askopenfilename(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))

    path, filename, filename_noext, extension, folder_path = Find_Experiments(path_filename)

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def ArrayCoord():

    global path_filename, path, filename, filename_noext, extension

    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes=[("CSV files", "*.csv")])

    if path_filenames == "":
        return

    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        txt.insert(END, "Analysing file {}...\n".format(filename))
        Spacer()
        try:
            df = pd.read_table(path_filename, encoding="iso-8859-1", low_memory=False,
                                   delim_whitespace=True, names=("X mean [um], Y mean [um]"))
            df = df.dropna(how='all')  # to drop if all values in the row are nan
            df = df.astype(float)
        except:
            df = pd.read_csv(path_filename)
            pass

        #Get columns from df
        X_col = df.columns.get_loc("X mean [um]")
        Y_col = df.columns.get_loc("Y mean [um]")

        #Define array of data
        X = np.array(df.iloc[:, X_col])
        Y = np.array(df.iloc[:, Y_col])

        X_list = sorted(X)
        n = len(X_list)

        with open("{}/Array Coordinates.csv".format(path), 'w', newline='') as f:
            Labels_list = ['X', 'Y']
            write = csv.writer(f)
            write.writerow(Labels_list)

            for i in range(n):
                PosX_new = round((X[i] - X[0]),1)
                PosY_new = round((Y[i] - Y[0]),1)

                Results_list = []
                Results_list.append([PosX_new, PosY_new])
                write.writerows(Results_list)

            Spacer()
            txt.insert(END, "Coordinates saved in: {}/Array Coordinates.csv\n".format(path))
            Spacer()
            Spacer()
            txt.insert(END, "END OF PROGRAM.\n".format(year_now))
            Spacer()
            Spacer()
            txt.update()
            txt.see("end")

            return PosX_new, PosY_new

#INTERFACE
l1 = Label(root, text="B.I.A.S.: Berkovich Indentation Analysis System", font='Helvetica 24 bold', fg = "SteelBlue4").grid(row = 0, column = 0, sticky = W, padx = 200, pady = 2)

l3 = Label(root, text="Experimental and fitting parameters:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 5, pady = 2)

l4 = Label(root, text="C0:").grid(row = 3, column = 0, sticky = W, padx = 5, pady = 2)
e2 = Entry(root, width=10)
e2.insert(END, "24.90306117772971")
e2.grid(row = 3, column = 0, sticky = W, padx = 350, pady = 2)
l5 = Label(root, text="C1:").grid(row = 4, column = 0, sticky = W, padx = 5, pady = 2)
e3 = Entry(root, width=10)
e3.insert(END, "-29.24962328386323")
e3.grid(row = 4, column = 0, sticky = W, padx = 350, pady = 2)
l6 = Label(root, text="C2:").grid(row = 5, column = 0, sticky = W, padx = 5, pady = 2)
e4 = Entry(root, width=10)
e4.insert(END, "56.89290759598595")
e4.grid(row = 5, column = 0, sticky = W, padx = 350, pady = 2)
l7 = Label(root, text="C3:").grid(row = 6, column = 0, sticky = W, padx = 5, pady = 2)
e5 = Entry(root, width=10)
e5.insert(END, "-67.78827859030005")
e5.grid(row = 6, column = 0, sticky = W, padx = 350, pady = 2)
l8 = Label(root, text="C4:").grid(row = 7, column = 0, sticky = W, padx = 5, pady = 2)
e6 = Entry(root, width=10)
e6.insert(END, "43.90789163565514")
e6.grid(row = 7, column = 0, sticky = W, padx = 350, pady = 2)
l9 = Label(root, text="C5:").grid(row = 8, column = 0, sticky = W, padx = 5, pady = 2)
e7 = Entry(root, width=10)
e7.insert(END, "-11.83187556909376")
e7.grid(row = 8, column = 0, sticky = W, padx = 350, pady = 2)
l10 = Label(root, text="Beyeler Fit Depth [um]:").grid(row = 9, column = 0, sticky = W, padx = 5, pady = 2)
e8 = Entry(root, width=5)
e8.insert(END, "0.1")
e8.grid(row = 9, column = 0, sticky = W, padx = 350, pady = 2)
l11 = Label(root, text="Beta:").grid(row = 10, column = 0, sticky = W, padx = 5, pady = 2)
e9 = Entry(root, width=5)
e9.insert(END, "1.01")
e9.grid(row = 10, column = 0, sticky = W, padx = 350, pady = 2)

l12 = Label(root, text="Sample's Poisson's ratio:").grid(row = 11, column = 0, sticky = W, padx = 5, pady = 2)
e10 = Entry(root, width=5)
e10.insert(END, "0.5") # Poisson's ratio of sample (v_PDMS 0.45-0.5; v_polystyrene 0.34)
e10.grid(row = 11, column = 0, sticky = W, padx = 350, pady = 2)

l13 = Label(root, text="Data Processing:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 12, column = 0, sticky = W, padx = 5, pady = 2)
b1 = Button(root, text="Extract data", command=File_Extractor_b1).grid(row = 13, column = 0, sticky = W, padx = 5, pady = 2)
b2 = Button(root, text="Divide Data", command=Gen_Single_Exp_b2).grid(row = 13, column = 0, sticky = W, padx = 130, pady = 2)
b3 = Button(root, text="Data to csv", command=Divide_Array_Data_b3).grid(row = 13, column = 0, sticky = W, padx = 260, pady = 2)

l14 = Label(root, text="Berkovich analysis:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 14, column = 0, sticky = W, padx = 5, pady = 2)
l15 = Label(root, text="Select processing model:").grid(row = 15, column = 0, sticky = W, padx = 5, pady = 2)
options = ["Find Contact", "No Contact"]
clicked = StringVar()
clicked.set(options[0])
dm1 = OptionMenu(root, clicked, *options).grid(row = 15, column = 0, sticky = W, padx = 150, pady = 2)
b4 = Button(root, text="Analyse individual measurements", command=BerkovichAnalysis_Individual_b4).grid(row = 16, column = 0, sticky = W, padx = 5, pady = 2)
b5 = Button(root, text="Analyse array measurements", command=BerkovichAnalysis_Array_b5).grid(row = 17, column = 0, sticky = W, padx = 5, pady = 2)
b6=Button(root, text="Get Array Coordinates", command=ArrayCoord).grid(row = 18, column = 0, sticky = W, padx = 5, pady = 2)

#Create and write inside a dialog box
l16 = Label(root, text="Dialog window:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 450, pady = 2)
txt = scrolledtext.ScrolledText(root, height=30, width=95)
txt.configure(font=("TkDefaultFont", 12, "normal"))
txt.grid(row=2, column = 0, rowspan = 17, sticky=W, padx = 450) #W=allign to left
txt.see("end")

root.mainloop()