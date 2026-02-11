import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def gru_cell_forward(xt, h_prev, parameters):
    """
    Arguments:
    xt -- input data at timestep "t", numpy array (n_x, m)
    h_prev -- Hidden state at timestep "t-1", numpy array (n_h, m)
    parameters -- python dictionary containing:
                  Wz, Uz, bz (Update Gate weights)
                  Wr, Ur, br (Reset Gate weights)
                  Wh, Uh, bh (Candidate Hidden weights)
    
    Returns:
    h_next -- next hidden state, of shape (n_h, m)
    """
    
    # Retrieve parameters
    Wz, Uz, bz = parameters["Wz"], parameters["Uz"], parameters["bz"]
    Wr, Ur, br = parameters["Wr"], parameters["Ur"], parameters["br"]
    Wh, Uh, bh = parameters["Wh"], parameters["Uh"], parameters["bh"]
    
    # --- 1. Update Gate (z) ---
    # Decides how much past info to keep
    # Equation: z_t = σ(Wz⋅x_t + Uz⋅h_{t-1} + bz)
    zt = sigmoid(np.dot(Wz, xt) + np.dot(Uz, h_prev) + bz)
    
    # --- 2. Reset Gate (r) ---
    # Decides how much past info to ignore
    # Equation: r_t = σ(Wr⋅x_t + Ur⋅h_{t-1} + br)
    rt = sigmoid(np.dot(Wr, xt) + np.dot(Ur, h_prev) + br)
    
    # --- 3. Candidate Hidden State (h_tilde) ---
    # Note the use of (rt * h_prev) -- applying the reset gate!
    # Equation: h~ = tanh(Wh⋅x_t + Uh⋅(r_t * h_{t-1}) + bh)
    h_candidate = tanh(np.dot(Wh, xt) + np.dot(Uh, (rt * h_prev)) + bh)
    
    # --- 4. Final Hidden State (h) ---
    # Interpolation between old state and new candidate
    # Equation: h_t = (1 - z_t) * h_{t-1} + z_t * h~
    h_next = (1 - zt) * h_prev + zt * h_candidate
    
    return h_next

# --- Quick Test ---
np.random.seed(1)
xt_tmp = np.random.randn(3, 10) # 3 features, batch of 10
h_prev_tmp = np.random.randn(5, 10) # 5 hidden units, batch of 10
params = {
    "Wz": np.random.randn(5, 3), "Uz": np.random.randn(5, 5), "bz": np.random.randn(5, 1),
    "Wr": np.random.randn(5, 3), "Ur": np.random.randn(5, 5), "br": np.random.randn(5, 1),
    "Wh": np.random.randn(5, 3), "Uh": np.random.randn(5, 5), "bh": np.random.randn(5, 1)
}

h_next = gru_cell_forward(xt_tmp, h_prev_tmp, params)
print("Next Hidden State Shape:", h_next.shape) # Should be (5, 10)