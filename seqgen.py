#!/usr/bin/python3

import numpy as np 


# Function to generate sequences of different lengths for the copy task
def generate_patterns(batch_size=100,
                      max_sequence=20,
                      min_sequence=1,
                      in_bits=8,
                      out_bits=8,
                      pad=1e-12,
                      low_tol=1e-12,
                      high_tol=1.0,
                      fixed_seq_len=False):  # Function to generate sequences of different lengths
    
    ti = []
    to = []

    for _ in range(batch_size):
        
        if not fixed_seq_len:
            seq_len_row = np.random.randint(low=min_sequence, high=max_sequence+1)
        else:
            seq_len_row = max_sequence

        pat = np.random.randint(low=0, high=2, size=(seq_len_row,out_bits))
        pat = pat.astype(np.float32)

        # Applying tolerance (So that values don't go to zero and cause NaN errors)
        pat[pat < 1] = low_tol
        pat[pat >= 1] = high_tol

        # Padding can be added if needed
        x = np.ones(((max_sequence*2)+2, in_bits+2), dtype=pat.dtype) * pad  # Input pattern has two extra side track
        y = np.ones(((max_sequence*2)+2, out_bits), dtype=pat.dtype) * pad  # Side tracks are not produced

        # Creates a delayed output (Target delay)
        x[1:seq_len_row+1, 2:] = pat
        y[seq_len_row+2:(2*seq_len_row)+2, :] = pat  # No side tracks needed for the output

        x[1:seq_len_row+1, 0:2] = low_tol
        x[0, :] = low_tol
        x[0, 1] = 1.0  # Start of sequence
        x[seq_len_row+1, :] = low_tol
        x[seq_len_row+1, 0] = 1.0  # End of sequence

        ti.append(x)
        to.append(y)

    return np.array(ti), np.array(to)
