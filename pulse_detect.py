# -*- coding: utf-8 -*-
"""
Created on Sat March 22 12:15:58 2025

@author: Amit Deokar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pywt
import os
import sys
import argparse
import pytz
from scipy.signal import savgol_filter
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="On-Pulse Dedispersed Gating")
    parser.add_argument('-gpt', type=str, help='GPT File')
    parser.add_argument('-ahdr', type=str, help='AHDR File')
    parser.add_argument('-dmp', type=str, help='Text File Containing DM and Period Info')
    parser.add_argument('-g_mode',type=int, default=-1, help='Gate End Phas Writing Mode: Set -1 for as it is and 1 for writing 0.5 if dedispered end gate phase is less than 0.5 and if greater than 0.5 set the value')
    parser.add_argument('-e', type=float, default = 0, help='Percentage Expansion of Bin window')
    return vars(parser.parse_args())

def merge_ranges(ranges, buffer=5):
    # Sort ranges by the start of each range
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    merged_ranges = []
    current_start, current_end = sorted_ranges[0]
    
    for start, end in sorted_ranges[1:]:
        # Check if ranges overlap or are within the buffer distance
        if start <= current_end + buffer:
            # Merge the ranges by extending the current range
            current_end = max(current_end, end)
        else:
            # If they do not overlap, add the previous range and start a new one
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last range
    merged_ranges.append((current_start, current_end))
    
    return merged_ranges


def pulse_region(signal):
    # Perform wavelet transform
    coeffs = pywt.wavedec(signal, 'db3', level=3)
    
    # Adaptive thresholding using local noise estimation
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(3 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    # Reconstruct signal from thresholded coefficients
    reconstructed_signal = pywt.waverec(coeffs, 'db3')
    
    # Apply Savitzky-Golay filter for smoothing
    smoothed_signal = savgol_filter(reconstructed_signal, window_length=51, polyorder=1)
    
    # Edge Detection using first derivative (gradient)
    gradient = np.gradient(smoothed_signal)
    
    # Find zero crossings in the gradient (where signal changes direction)
    zero_crossings = np.where(np.diff(np.sign(gradient)))[0]
    
    # Identify pulse regions using the smoothed signal and zero crossings
    pulse_regions = []
    for i in range(len(zero_crossings) - 1):
        start = zero_crossings[i]
        end = zero_crossings[i + 1]
        if np.max(smoothed_signal[start:end]) > threshold:
            pulse_regions.append((start, end))
    
    # Trim or adjust pulse regions as needed
    trimmed_pulse_regions = []
    for reg in pulse_regions:
        start, end = reg
        if smoothed_signal[start:end].max() > threshold:
            trimmed_pulse_regions.append((start, end))
    
    # Identify off-pulse regions
    off_pulse_ranges = []
    if trimmed_pulse_regions[0][0] > 0:
        off_pulse_ranges.append((0, trimmed_pulse_regions[0][0] - 1))
    for i in range(len(trimmed_pulse_regions) - 1):
        off_pulse_ranges.append((trimmed_pulse_regions[i][1] + 1, trimmed_pulse_regions[i + 1][0] - 1))
    if trimmed_pulse_regions[-1][1] < len(signal) - 1:
        off_pulse_ranges.append((trimmed_pulse_regions[-1][1] + 1, len(signal) - 1))
    
    return off_pulse_ranges, trimmed_pulse_regions

def read_ahdr(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            result_dict[key.strip()] = value.strip()
    return result_dict

def read_dm_pp(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == '':
                pass
            else:
                key, value = line.split(': ')
                value = ''.join(filter(lambda x: x.isdigit() or x == '.', value))  # Remove non-numeric characters
                result_dict[key.strip()] = value.strip()
    return result_dict

def ist_to_unix(date_str, time_str):
    ns_part = time_str[-3:]
    time_str = time_str[:-3]
    # Parse the date string into a datetime object
    local_dt = datetime.strptime(f'{date_str} {time_str}', '%d/%m/%Y %H:%M:%S.%f')

    # Specify the IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    
    # Localize the datetime object to IST
    local_dt = ist.localize(local_dt)
    
    # Convert the localized datetime to Unix time (seconds since the epoch)
    unix_time = local_dt.timestamp()
    unix_time_ns = f'{unix_time}{ns_part}'
    
    return unix_time_ns

def new_ist(date_str, time_str, delta):
    ns_part = time_str[-3:]
    time_str = time_str[:-3]
    
    local_dt = datetime.strptime(f'{date_str} {time_str}', '%d/%m/%Y %H:%M:%S.%f')

    # Specify the IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    
    # Localize the datetime object to IST
    local_dt = ist.localize(local_dt)
    
    # Create a timedelta object for milliseconds
    time_delta = timedelta(milliseconds=delta)
    
    # Add or subtract milliseconds from the datetime object
    new_dt = local_dt + time_delta
    
    # Format the new datetime in dd/mm/yyyy hh:mm:ss.ssssss format
    formatted_new_dt = new_dt.strftime('%d/%m/%Y %H:%M:%S.%f')
    new_date_str, new_time_str = formatted_new_dt.split(' ')    
    new_time_str = f'{new_time_str}{ns_part}'
    
    return new_date_str, new_time_str


def out_txt(pp, n_unix_time, on_gat_len, args):
    pp_s_n = "{:.9f}".format(pp*1e-3)
    o_txt = f'{pp_s_n}\n0.00000000000\n{n_unix_time}\n{on_gat_len}\n'
    dir_path = os.path.dirname(args['gpt'])
    file_name = os.path.basename(args['ahdr'])
    if dir_path == '':
        dir_path = '.'
    with open(f'{dir_path}/{file_name}.pulsar_period.txt','w') as of:
        of.write(o_txt)
        
def edge_pulse_shift(sig):
    max_i = np.argmax(sig)
    shift = (len(sig) // 2) - max_i
    cent_sig = np.roll(sig,shift)
    return cent_sig, shift

def edge_pulse_shift_inv(sig,shift,pulse_reg):
    sig = np.roll(sig,-shift)
    pulse_reg = (pulse_reg - shift)%len(sig)
    return sig, pulse_reg

def find_off_pul_reg(total_reg, pulse_reg):
    # Ensure pulse_reg is sorted based on the start of each interval
    pul_reg_c = pulse_reg.copy()
    pul_reg_c.sort()

    off_pulse_reg = []
    total_start, total_end = total_reg

    # 1. Handle the first "off pulse" region before the first pulse interval
    if pul_reg_c[0][0] > total_start:
        off_pulse_reg.append((total_start, pul_reg_c[0][0]))

    # 2. Handle "off pulse" regions between consecutive pulse intervals
    for i in range(1, len(pul_reg_c)):
        prev_end = pul_reg_c[i - 1][1]
        curr_start = pul_reg_c[i][0]
        if curr_start > prev_end:
            off_pulse_reg.append((prev_end, curr_start))

    # 3. Handle the last "off pulse" region after the last pulse interval
    if pul_reg_c[-1][1] < total_end:
        off_pulse_reg.append((pul_reg_c[-1][1], total_end))
    
    trimmed_off_pul_reg = []
    for reg in off_pulse_reg:
        diff = reg[1] - reg[0]
        trim_amt = int(diff * 0.1)
        trimmed_off_pul_reg.append((reg[0]+trim_amt,reg[1]-trim_amt))
        
    return trimmed_off_pul_reg

def on_edge(pulse_reg,csft,y):
    p_reg_t = [tuple(pulse_reg[0]-csft)]
    
    if (p_reg_t[0][0] >= 0) and (p_reg_t[0][1] <= len(y)):
        print('Pulse inside the Boundary')
        return False
    else:
        print('Pulse at Edge')
        return True

def on_key(event):
    """Handles key press events."""
    global left_boundary, right_boundary, active_listener
    
    leg.remove()
    fig1.canvas.draw()   
        
    if event.key == 'l':  # Press 'L' to set the left boundary
        while spans:
            span = spans.pop()  # Remove the last added span
            span.remove()
            fig1.canvas.draw()
        print("Click to set the LEFT boundary.")
        if active_listener is not None:
            fig1.canvas.mpl_disconnect(active_listener)  # Remove previous listener
        active_listener = fig1.canvas.mpl_connect('button_press_event', on_left_click)

    elif event.key == 'r':  # Press 'R' to set the right boundary
        while spans:
            span = spans.pop()  # Remove the last added span
            span.remove()
            fig1.canvas.draw()
        print("Click to set the RIGHT boundary.")
        if active_listener is not None:
            fig1.canvas.mpl_disconnect(active_listener)  # Remove previous listener
        active_listener = fig1.canvas.mpl_connect('button_press_event', on_right_click)

    elif event.key == 'a':  # Press 'A' to accept the selection and close
        if left_boundary is not None and right_boundary is not None:
            print(f"Accepted: Left = {left_boundary}, Right = {right_boundary}")
            plt.close(fig1)
            main(left_boundary, right_boundary)
        else:
            print("Please select both left and right boundaries before accepting.")


def on_left_click(event):
    """Handles selection of the left boundary."""
    global left_boundary, left_line, right_line
    if event.button == 1:  # Right-click only
        left_boundary = event.xdata
        print(f"Left boundary set at: {left_boundary}")

        # Remove previous left boundary line if it exists
        if left_line is not None:
            left_line.remove()

        # Draw new left boundary line
        left_line = ax21.axvline(left_boundary, color='r', linestyle='dashed')

        # Keep the right boundary visible if already set
        if right_boundary is not None:
            if right_line is not None:
                right_line.remove()
            right_line = ax21.axvline(right_boundary, color='b', linestyle='dashed')

        fig1.canvas.draw()


def on_right_click(event):
    """Handles selection of the right boundary."""
    global right_boundary, right_line, left_line
    if event.button == 1:  # Right-click only
        right_boundary = event.xdata
        print(f"Right boundary set at: {right_boundary}")

        # Remove previous right boundary line if it exists
        if right_line is not None:
            right_line.remove()

        # Draw new right boundary line
        right_line = ax21.axvline(right_boundary, color='b', linestyle='dashed')

        # Keep the left boundary visible if already set
        if left_boundary is not None:
            if left_line is not None:
                left_line.remove()
            left_line = ax21.axvline(left_boundary, color='r', linestyle='dashed')

        fig1.canvas.draw()
    

def main(left, right):
    print('\n**************On-Pulse Dedispersed Gating**************\n')
    global fig1, ax21, spans, leg
    
    if left is not None and right is not None:
        print('Code re-run using manually defined gates')
    args = parse_args()
    txt = np.loadtxt(args['gpt'], delimiter=' ')
    ahdr = read_ahdr(args['ahdr'])
    dm_pp = read_dm_pp(args['dmp'])
    
    chan0 = float(ahdr['Frequency Ch.1  (Hz)'])/1e9
    bwd = float(ahdr['Bandwidth (MHz)'])/1e3
    nch = int(ahdr['Channels'])
    chn_wid_hz = float(ahdr['Channel width (Hz)'])
    st = float(ahdr['Sampling time  (uSec)'])/1e3
    time0 = ahdr['IST Time']
    date_ob = ahdr['Date']
    dm = float(dm_pp['Dispersion measure'])
    pp = float(dm_pp['Pulsar period'])
    sb = np.sign(float(ahdr['Channel width (Hz)']))
    
    print(f'Observation Time (IST): {date_ob} {time0}')
    
    chanN = chan0 + (sb * bwd)
    if chanN > chan0:
        ref_freq = chanN
        freq = chan0
    elif chan0 > chanN:
        ref_freq = chan0
        freq = chanN
    
    delays_ms = 4.15 * dm * ((1 / freq**2) - (1 / ref_freq**2))
    print(f'Dispersion delay: {delays_ms} ms')
    
    if delays_ms > 0.8 * pp:
        wrn_cnt = input(f'\nWarning: Despersion delay is {round(delays_ms/pp*100,2)}% of the pulsar period.\nDo you want to continue (Y/N)?: ')
        if wrn_cnt.upper() == 'Y':
            pass
        else:
            sys.exit()
        
    unix_time0 = ist_to_unix(date_ob, time0)
    
    x = txt[:,0]
    y = txt[:,1]
    
    # y = np.roll(y, shift=1000)
    
    if left is not None and right is not None:
        y = y - np.median(y)
        y = y / max(y)
        
        csft = 0
        if left < right:
            is_on_edge = False
        if left > right:
            is_on_edge = True
        
        if is_on_edge:
            print('Pulse at Edge')
            pulse_reg = np.array([[left/st, x[-1] + right/st]])
        else:
            print('Pulse inside the Boundary')
            pulse_reg = np.array([[left/st, right/st]])
            
        diffs = [y - x for x, y in pulse_reg]
    else:
        # #############################################
        # # if args['ecorr'] == 1:
        y, csft = edge_pulse_shift(y)
        # #############################################
        
        y = y - np.median(y)
        y = y / max(y)
        
        off_pul_reg, pulse_reg0 = pulse_region(signal=y)
        
        pulse_reg1 = [merge_ranges(pulse_reg0)[0]]
        
        pulse_reg = []
        for reg in pulse_reg1:
                diff = reg[1] - reg[0]
                exp_amt = int(diff * (args['e']/100))
                pulse_reg.append((reg[0]-exp_amt,reg[1]+exp_amt))
                
        is_on_edge = on_edge(pulse_reg,csft,y)
        edge_mode = is_on_edge
        
        # # off_pul_reg = find_off_pul_reg((0,len(y)), pulse_reg)    
        
        diffs = [y - x for x, y in pulse_reg]
        
        # # max_sig = max(y[pulse_reg[np.argmax(diffs)][0]:pulse_reg[np.argmax(diffs)][1]])
        # # rng_mn = []
        # # for of in off_pul_reg:
        # #     rng_mn.append(np.mean(y[of[0]:of[1]]))
        # # of_mn = np.mean(rng_mn)
        
        # # snr = abs((max_sig - of_mn)/of_mn)
        # # print(f'SNR: {snr}')
        
        ###########################################################
        y, pulse_reg = edge_pulse_shift_inv(y, csft, pulse_reg)
        # print(f'On-Pulse Region: {pulse_reg}')
        ###########################################################
    
    time_shift_ms = (pulse_reg[np.argmax(diffs)][0]*st)
    print(f'\nOn-Gate Start Time: {time_shift_ms} ms')
    
    n_dt_ist, n_tm_ist = new_ist(date_ob, time0, time_shift_ms)
    # print(f'New IST: {n_dt_ist} {n_tm_ist}')
    n_unix_time = ist_to_unix(n_dt_ist, n_tm_ist)
    
    sft = int(delays_ms/st)
    
    fig1, ax21 = plt.subplots()
    x = np.arange(len(y)) * st

    ax21.plot(x,y)
    ax21.set_ylabel('Power (in arbitary units)')
    ax21.set_xlabel('Time (in ms)')

    for i in range(len(pulse_reg)):
        g0 = (pulse_reg[i][0]*st)%x[-1]
        g1 = (pulse_reg[i][1]*st)%x[-1]
        g2 = ((pulse_reg[i][0]*st)+delays_ms)%x[-1]
        g3 = ((pulse_reg[i][1]*st)+delays_ms)%x[-1]
        
        spans = []
        
        if is_on_edge == False:
            spans.append(ax21.axvspan(g0,g1, color='red', alpha=0.1, label='On-Pulse (High-Frequency)'))
        
            if (((pulse_reg[i][0]*st)+delays_ms) < x[-1]) and (((pulse_reg[i][1]*st)+delays_ms) > x[-1]):
                spans.append(ax21.axvspan(g2,x[-1], color='green', alpha=0.1, label='On-Pulse (Low-Frequency)'))
                spans.append(ax21.axvspan(0,g3, color='green', alpha=0.1))
            else:
                spans.append(ax21.axvspan(g2,g3, color='green', alpha=0.1, label='On-Pulse (Low-Frequency)'))
        if is_on_edge:
            spans.append(ax21.axvspan(g0,x[-1], color='red', alpha=0.1, label='On-Pulse (High-Frequency)'))
            spans.append(ax21.axvspan(0,g1, color='red', alpha=0.1))
            
            if g2 > g3:
                spans.append(ax21.axvspan(g2,x[-1], color='green', alpha=0.1, label='On-Pulse (Low-Frequency)'))
                spans.append(ax21.axvspan(0,g3, color='green', alpha=0.1, label='On-Pulse (Low-Frequency)'))
            else:
                spans.append(ax21.axvspan(g2,g3, color='green', alpha=0.1, label='On-Pulse (Low-Frequency)'))

        if i == 0:
            leg = ax21.legend()
    ax21.set_title("Detected Pulse Region\nManual Mode: 'L' for Left, 'R' for Right, 'A' to Accept (Click to Select)")
    if is_on_edge:
        gate_len = ((x[-1] - g0) + g3)/pp
    else:
        gate_len = (((pulse_reg[np.argmax(diffs)][1]*st)+delays_ms) - (time_shift_ms))/pp
        
    if args['g_mode'] == 1:
        if gate_len <= 0.5:
            ogl = 0.5
        else:
            ogl = gate_len
    if args['g_mode'] == -1:
        ogl = round(gate_len, 3)
    
    print(f'On-Gate Length: {ogl}\n')
    out_txt(pp, n_unix_time, ogl, args)


if __name__ == "__main__":
    # Disable 'L' key's default zoom behavior
    mpl.rcParams['keymap.xscale'] = []
    mpl.rcParams['keymap.yscale'] = []
    mpl.rcParams['keymap.home'] = []

    # Global variables to store boundaries and lines
    left_boundary = None
    right_boundary = None
    left_line = None
    right_line = None
    active_listener = None  # Stores active event listener ID
    running = True  # Flag to control the loop

    while running:
        main(left=left_boundary, right=right_boundary)

        fig1.canvas.mpl_connect('key_press_event', on_key)  # Connect the key event handler

        print("\nPress 'q' to quit or close the figure to rerun.")

        plt.show(block=False)  # Show the plot without blocking the loop
        while plt.fignum_exists(fig1.number):  # Check if the figure is open
            plt.pause(0.1)  # Small pause to allow GUI updates

        # Ask the user if they want to exit or rerun
        user_input = input("Press 'q' to quit or Enter to re-run: ").strip().lower()
        if user_input == 'q':
            running = False  # Exit the loop
