## Imports used for this lab
import numpy as np
import matplotlib.pyplot as plt
import os


## Task 1: Distance Estimation from Phase Measurements


## Task 2: Angle of Arrival Estimation

def estimate_aoa(h, d, wavelength):
    """
    Estimate the Angle of Arrival (AoA) of a reflected signal from RFID tag.
    Parameters:
        h : np.ndarray
            Channel frequency response (CFR) vector.
        d : float
            Distance between antenna elements in meters.
        wavelength : float
            Wavelength of the signal in meters.
    Returns:
        aoa : float
            Estimated angle of arrival in degrees.
    """

    N = len(h)
    theta_scan = np.arange(0, 181, 1)   # resolution of 1 degree
    response = np.zeros(len(theta_scan), dtype=np.complex128)

    for theta_deg, theta in enumerate(theta_scan):
        theta_rad = np.deg2rad(theta)
        steering = np.exp(-1j * (np.arange(N) + 1) * 4 * np.pi / wavelength * d * np.cos(theta_rad))    # + 1 s.t. i = 1,..,N
        response[theta_deg] = np.abs(np.sum(np.exp(1j * np.deg2rad(h)) * steering))

    aoa = theta_scan[np.argmax(response)]
    return aoa

def estimate_aoa_circular(h, R, wavelength):
    """"
    Estimate the Angle of Arrival (AoA) of a reflected signal from RFID tag using circular array.
    Parameters:
        h : np.ndarray
            Channel frequency response (CFR) vector.
        R : float
            Radius of the circular array in meters.
        wavelength : float
            Wavelength of the signal in meters.
    Returns:
        aoa : float
            Estimated angle of arrival in degrees.
    """

    N = len(h)
    theta_scan = np.arange(0, 360, 1)   # resolution of 1 degree
    response = np.zeros(len(theta_scan), dtype=np.complex128)

    phi = np.arange(N) * 2 * np.pi / N

    for theta_deg, theta in enumerate(theta_scan):
        theta_rad = np.deg2rad(theta)
        steering = np.exp(1j * 4 * np.pi / wavelength * R * np.cos(phi - theta_rad))
        response[theta_deg] = np.abs(np.sum(np.exp(1j * np.deg2rad(h)) * steering))

    aoa = theta_scan[np.argmax(response)]
    return aoa


## Task 3: Time of Flight Estimation

def spatial_beam_pattern(x, y, antenna_pos, h, wavelength):
    """
    Compute the spatial beam pattern at position (x, y) given antenna positions.
    Parameters:
        x : float
            x-coordinate of the point where the beam pattern is evaluated.
        y : float
            y-coordinate of the point where the beam pattern is evaluated.
        antenna_pos : np.ndarray
            Array of shape (N, 2) containing the (x, y) positions of N antennas.
        h : np.ndarray
            Array of shape (N,) containing the phase shifts for each antenna.
        wavelength : float
            Wavelength of the signal.
    Returns:
        power : float
            The computed power of the beam pattern at (x, y).
    """
    
    xi = antenna_pos[:, 0]
    yi = antenna_pos[:, 1]

    ri = np.sqrt((x - xi)**2 + (y - yi)**2)
    steering = np.exp(1j * 4 * np.pi / wavelength * ri)
    signal = np.exp(-1j * h) * steering

    power = np.abs(np.sum(signal))**2

    return power

def get_source_channel(antenna_coord, source, wavelength):
    """
    Compute the channel phases from a source to each antenna.
    Parameters:
        antenna_coord : np.ndarray
            Array of shape (N, 2) containing the (x, y) positions of N antennas.
        source : tuple
            Tuple (sx, sy) representing the (x, y) position of the source.
        wavelength : float
            Wavelength of the signal.
    Returns:
        channels : np.ndarray
            Array of shape (N,) containing the channel phases for each antenna.
    """
    xs = antenna_coord[:, 0]
    ys = antenna_coord[:, 1]
    sx, sy = source

    d = np.sqrt((sx - xs)**2 + (sy - ys)**2)
    channels = 4 * np.pi / wavelength * d

    return channels

def make_ant_pos(antenna_spacing, num_ant):
    """
    Generate antenna positions in a linear array centered at the origin.
    Parameters:
        antenna_spacing : float
            Spacing between adjacent antennas.
        num_ant : int
            Number of antennas in the array.
    Returns:
        antenna_coord : np.ndarray
            Array of shape (N, 2) containing the (x, y) positions of N antennas.
    """
    xs = np.arange(num_ant) * antenna_spacing
    xs = xs - np.mean(xs)
    ys = np.zeros(num_ant)
    antenna_coord = np.column_stack((xs, ys))

    return antenna_coord

## Task 3: (part 4.2) Testing on Data from RFID Hardware

### Begin of given code
# Values of h1, h2 and h3 provided here
h1 = 66.2000  # In degrees
h2 = 109.6875  # In degrees
h3 = 164.4250  # In degrees

# Convert to radians
h1 = np.pi * h1 / 180
h2 = np.pi * h2 / 180
h3 = np.pi * h3 / 180

# Coordinates of Antenna Elements in 2D plane provided here
# Assume ant 1 is at origin and all other antennas lie on x axis
d = 7.62e-2
ant1 = np.array([0, 0])
ant2 = np.array([-d, 0])
ant3 = np.array([-2 * d, 0])

# Frequency and Lambda provided here
freq = 9.0275e8
wavelength = 3e8 / freq
### End of given code

x_grid = np.linspace(-1, 1, 400)
y_grid = np.linspace(0, 2, 400)

def compute_pattern(antenna_pos, h, wavelength):
    M = np.zeros((len(x_grid), len(y_grid)))

    for j in range(len(x_grid)):
        for k in range(len(y_grid)):
            M[j, k] = spatial_beam_pattern(x_grid[j], y_grid[k], antenna_pos, h, wavelength)

    M = np.transpose(M)
    return M

# Antenna 1 and Antenna 2
antenna_pos_12 = np.vstack([ant1, ant2])
h_12 = np.array([h1, h2])
M12 = compute_pattern(antenna_pos_12, h_12, wavelength)

plt.figure(figsize=(7,6))
plt.imshow(M12, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', cmap='gray', aspect='auto')
plt.title('Spatial beam pattern - Antenna 1 & 2')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.colorbar()
plt.show()

# Antenna 1 and Antenna 3
antenna_pos_13 = np.vstack([ant1, ant3])
h_13 = np.array([h1, h3])
M13 = compute_pattern(antenna_pos_13, h_13, wavelength)

plt.figure(figsize=(7,6))
plt.imshow(M13, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', cmap='gray', aspect='auto')
plt.title('Spatial beam pattern - Antenna 1 & 3')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.colorbar()
plt.show()

## Task 4: Localization Using Multiple RFID Tags
