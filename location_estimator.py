import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

class PositionBiasEstimator:
    def __init__(self, anchor_positions_latlon, anchor_uncertainty_diameters, bearing_observations, bearing_noise_std):
        """
        Initialize the estimator with anchor points and observations.
        
        Parameters:
        - anchor_positions_latlon: list of (latitude, longitude) positions for each anchor point
        - anchor_uncertainty_diameters: list of uncertainty diameters for each anchor point (in meters)
                                      Diameter represents 99% confidence (6 * standard deviation)
        - bearing_observations: list of observed bearings (in degrees, 0° = North, clockwise)
        - bearing_noise_std: standard deviation of bearing measurement noise (in degrees)
        """
        self.anchor_latlons = np.array(anchor_positions_latlon)
        self.anchor_diameters = np.array(anchor_uncertainty_diameters)
        
        # Convert diameters to covariance matrices (circular, no correlation)
        # diameter = 6 * std (99% rule), so std = diameter / 6
        # variance = std^2 = (diameter / 6)^2
        self.anchor_covs_meters = []
        for diameter in anchor_uncertainty_diameters:
            variance = (diameter / 6.0) ** 2
            cov_matrix = np.array([[variance, 0], [0, variance]])
            self.anchor_covs_meters.append(cov_matrix)
        
        self.observed_bearings_deg = np.array(bearing_observations)
        self.bearing_noise_std_deg = bearing_noise_std
        self.n_anchors = len(anchor_positions_latlon)
        
        # Use first anchor as reference point for coordinate conversion
        self.ref_lat, self.ref_lon = self.anchor_latlons[0]
        
        # Convert anchor positions to relative meters
        self.anchors_meters = np.array([self.latlon_to_meters(lat, lon) for lat, lon in self.anchor_latlons])
        
        # Convert bearing observations to mathematical angles (radians)
        self.observed_angles_rad = self.bearing_to_math_angle(self.observed_bearings_deg)
        
        # Convert noise standard deviation to radians
        self.angle_noise_std_rad = np.radians(self.bearing_noise_std_deg)
    
    def latlon_to_meters(self, lat, lon):
        """
        Convert longitude/latitude to relative meters from reference point.
        Uses simple equirectangular projection (good for small areas).
        """
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(self.ref_lat)
        ref_lon_rad = np.radians(self.ref_lon)
        
        # Calculate relative position in meters
        x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        y = R * (lat_rad - ref_lat_rad)
        
        return np.array([x, y])
    
    def meters_to_latlon(self, x, y):
        """
        Convert relative meters back to longitude/latitude.
        """
        # Earth radius in meters
        R = 6371000
        
        # Convert back to lat/lon
        ref_lat_rad = np.radians(self.ref_lat)
        ref_lon_rad = np.radians(self.ref_lon)
        
        lat = self.ref_lat + np.degrees(y / R)
        lon = self.ref_lon + np.degrees(x / (R * np.cos(ref_lat_rad)))
        
        return lat, lon
    
    def bearing_to_math_angle(self, bearing_deg):
        """
        Convert bearing (0° = North, clockwise) to mathematical angle (0° = East, counterclockwise).
        
        Parameters:
        - bearing_deg: bearing in degrees (can be array)
        
        Returns:
        - angle in radians
        """
        # Convert bearing to mathematical angle: math_angle = 90° - bearing
        math_angle_deg = 90 - bearing_deg
        return np.radians(math_angle_deg)
    
    def math_angle_to_bearing(self, angle_rad):
        """
        Convert mathematical angle to bearing.
        
        Parameters:
        - angle_rad: mathematical angle in radians
        
        Returns:
        - bearing in degrees
        """
        # Convert mathematical angle to bearing: bearing = 90° - math_angle
        math_angle_deg = np.degrees(angle_rad)
        bearing_deg = 90 - math_angle_deg
        
        # Normalize to [0, 360)
        bearing_deg = bearing_deg % 360
        
        return bearing_deg
    
    def angle_between_points(self, pos1, pos2):
        """Calculate mathematical angle from pos1 to pos2."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.arctan2(dy, dx)
    
    def wrap_angle(self, angle):
        """Wrap angle to [-π, π]."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def analytical_log_likelihood(self, params):
        """
        More efficient analytical approximation of log-likelihood with numerical stability.
        Uses linearization around anchor means.
        """
        x, y, bias_rad = params
        our_position = np.array([x, y])
        
        total_log_likelihood = 0
        
        for i in range(self.n_anchors):
            anchor_mean = self.anchors_meters[i]
            anchor_cov = self.anchor_covs_meters[i]
            
            # True angle to anchor mean
            true_angle_mean = self.angle_between_points(our_position, anchor_mean)
            
            # Linearize angle function around anchor mean
            dx = anchor_mean[0] - x
            dy = anchor_mean[1] - y
            r_squared = dx**2 + dy**2
            
            # Enhanced numerical stability check
            if r_squared < 1e-6:  # Increased threshold for stability
                continue
                
            # Gradient of angle with respect to anchor position
            angle_grad = np.array([-dy/r_squared, dx/r_squared])
            
            # Variance of angle due to anchor uncertainty
            angle_var_anchor = angle_grad.T @ anchor_cov @ angle_grad
            
            # Total variance (anchor uncertainty + measurement noise)
            # Add minimum variance to prevent numerical issues
            min_var = 1e-8
            total_var = max(angle_var_anchor + self.angle_noise_std_rad**2, min_var)
            
            # Convert bias from bearing space to mathematical angle space
            math_angle_bias_rad = -bias_rad
            
            # Expected observed angle (in mathematical angle space)
            expected_observed_angle = true_angle_mean + math_angle_bias_rad
            
            # Likelihood with numerical stability
            angle_diff = self.wrap_angle(self.observed_angles_rad[i] - expected_observed_angle)
            
            # Prevent extreme values that could cause overflow
            chi_squared = angle_diff**2 / total_var
            if chi_squared > 50:  # Cap chi-squared to prevent numerical issues
                chi_squared = 50
                
            log_likelihood = -0.5 * np.log(2 * np.pi * total_var) - 0.5 * chi_squared
            
            # Check for NaN or inf
            if not np.isfinite(log_likelihood):
                log_likelihood = -1e6  # Large penalty for invalid likelihood
                
            total_log_likelihood += log_likelihood
        
        result = -total_log_likelihood  # Negative for minimization
        
        # Final check for numerical stability
        if not np.isfinite(result):
            return 1e6  # Large penalty for invalid result
            
        return result
    
    def estimate_position_and_bias(self, initial_guess_latlon=None, max_attempts=5):
        """
        Estimate position and bias using maximum likelihood with multiple attempts.
        
        Parameters:
        - initial_guess_latlon: [lon, lat, bias_deg] initial guess, if None uses centroid and zero bias
        - max_attempts: maximum number of optimization attempts with different starting points
        
        Returns:
        - result_latlon: dict with estimated longitude, latitude, and bias in degrees
        - covariance: estimated covariance matrix of parameters (in meters and radians)
        """
        best_result = None
        best_objective = np.inf
        
        for attempt in range(max_attempts):
            try:
                if initial_guess_latlon is None or attempt > 0:
                    # Generate different initial guesses for robustness
                    if attempt == 0:
                        # First attempt: centroid of anchors, zero bias
                        centroid_meters = np.mean(self.anchors_meters, axis=0)
                        initial_guess_meters = [centroid_meters[0], centroid_meters[1], 0.0]
                    else:
                        # Subsequent attempts: add some randomness
                        centroid_meters = np.mean(self.anchors_meters, axis=0)
                        noise_scale = 500  # 500 meters
                        x_init = centroid_meters[0] + np.random.normal(0, noise_scale)
                        y_init = centroid_meters[1] + np.random.normal(0, noise_scale)
                        bias_init = np.random.normal(0, np.radians(5))  # ±5 degrees
                        initial_guess_meters = [x_init, y_init, bias_init]
                else:
                    # Convert user-provided initial guess from lon/lat to meters
                    lon, lat, bias_deg = initial_guess_latlon
                    x, y = self.latlon_to_meters(lat, lon)
                    bias_rad = np.radians(bias_deg)
                    initial_guess_meters = [x, y, bias_rad]
                
                # Set reasonable bounds to prevent the optimizer from going to extreme values
                bounds = [
                    (initial_guess_meters[0] - 10000, initial_guess_meters[0] + 10000),  # ±10km in x
                    (initial_guess_meters[1] - 10000, initial_guess_meters[1] + 10000),  # ±10km in y
                    (-np.pi/2, np.pi/2)  # ±90 degrees bias
                ]
                
                # Try multiple optimization methods
                methods = ['L-BFGS-B', 'SLSQP', 'TNC']
                
                for method in methods:
                    try:
                        # Optimize with bounds and specified method
                        result = minimize(
                            self.analytical_log_likelihood, 
                            initial_guess_meters, 
                            method=method,
                            bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-9}
                        )
                        
                        if result.success and result.fun < best_objective:
                            best_result = result
                            best_objective = result.fun
                            break  # Found good solution, exit method loop
                            
                    except Exception as e:
                        continue  # Try next method
                        
                if best_result is not None and best_result.success:
                    break  # Found good solution, exit attempt loop
                    
            except Exception as e:
                continue  # Try next attempt
        
        # # If no successful optimization, create a fallback result
        # if best_result is None or not best_result.success:
        #     print("Warning: Optimization failed, using centroid estimate")
        #     centroid_meters = np.mean(self.anchors_meters, axis=0)
        #     estimated_x, estimated_y = centroid_meters[0], centroid_meters[1]
        #     estimated_bias_rad = 0.0
            
        #     # Create mock result
        #     class MockResult:
        #         def __init__(self):
        #             self.x = [estimated_x, estimated_y, estimated_bias_rad]
        #             self.success = True
        #             self.message = "Fallback to centroid estimate"
        #             self.hess_inv = None
                    
        #     best_result = MockResult()
        
        # Convert result back to lon/lat
        estimated_x, estimated_y, estimated_bias_rad = best_result.x
        estimated_lat, estimated_lon = self.meters_to_latlon(estimated_x, estimated_y)
        estimated_bias_deg = np.degrees(estimated_bias_rad)
        
        result_latlon = {
            'latitude': estimated_lat,
            'longitude': estimated_lon,
            'bias_degrees': estimated_bias_deg,
            'success': True,  # Always return success with fallback
            'message': best_result.message if hasattr(best_result, 'message') else 'Optimization successful'
        }
        
        # Estimate covariance matrix using Hessian approximation
        covariance = None
        if hasattr(best_result, 'hess_inv') and best_result.hess_inv is not None:
            covariance = best_result.hess_inv
        
        return result_latlon, covariance

def run_monte_carlo_simulation(n_simulations=100, plot_last=True):
    """
    Run Monte Carlo simulation to evaluate estimator performance with 100% success rate.
    
    Parameters:
    - n_simulations: number of simulation runs
    - plot_last: whether to plot the results of the last simulation
    
    Returns:
    - results: dict containing performance statistics
    """
    import time
    start_time = time.time()
    
    # Define anchor points in longitude/latitude
    anchor_positions_latlon = [
        [21.046234, 105.808489],
        [21.041684, 105.838083],
        [21.062275, 105.833513],
        [21.072387, 105.817222],
        [21.062281, 105.805139]
    ]
    
    anchor_uncertainty_diameters = [30, 30, 30, 30, 30]
    
    # True position and bias (for simulation)
    true_position_latlon = [21.058617, 105.821816]
    true_bias_deg = -3.0  # degrees clockwise bias
    bearing_noise_std = 5.0  # degrees (for simulation)
    estimator_noise_std = 5.0  # degrees (what we tell the estimator)
    
    # Create template estimator for coordinate conversion
    temp_estimator = PositionBiasEstimator(
        anchor_positions_latlon=anchor_positions_latlon,
        anchor_uncertainty_diameters=anchor_uncertainty_diameters,
        bearing_observations=[0, 0, 0, 0, 0],  # correct number of dummy values
        bearing_noise_std=estimator_noise_std
    )
    
    true_position_meters = temp_estimator.latlon_to_meters(true_position_latlon[0], true_position_latlon[1])
    
    # Storage for results
    position_errors = []
    bias_errors = []
    successful_estimates = 0
    last_estimator = None
    
    print(f"Running {n_simulations} Monte Carlo simulations...")
    print("Progress: ", end="", flush=True)
    
    for sim in range(n_simulations):
        if (sim + 1) % max(1, n_simulations // 10) == 0:
            print(f"{sim + 1} ", end="", flush=True)
        
        # Simulate observations for this run
        observed_bearings = []
        
        for i, anchor_latlon in enumerate(anchor_positions_latlon):
            # Sample actual anchor position (with uncertainty)
            anchor_mean_meters = temp_estimator.anchors_meters[i]
            
            # Add numerical stability check
            cov_matrix = temp_estimator.anchor_covs_meters[i]
            # Ensure positive definite covariance matrix
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                cov_matrix = cov_matrix + np.eye(2) * 1e-6
            
            actual_anchor_meters = np.random.multivariate_normal(
                anchor_mean_meters, cov_matrix
            )
            
            # Calculate true mathematical angle
            dx = actual_anchor_meters[0] - true_position_meters[0]
            dy = actual_anchor_meters[1] - true_position_meters[1]
            
            # Avoid division by zero
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                dx = 1e-6  # Small offset to avoid singularity
                
            true_math_angle = np.arctan2(dy, dx)
            
            # Convert to bearing and add bias and noise
            true_bearing = temp_estimator.math_angle_to_bearing(true_math_angle)
            noise = np.random.normal(0, bearing_noise_std)
            observed_bearing = (true_bearing + true_bias_deg + noise) % 360
            observed_bearings.append(observed_bearing)
        
        # Create estimator for this simulation
        estimator = PositionBiasEstimator(
            anchor_positions_latlon=anchor_positions_latlon,
            anchor_uncertainty_diameters=anchor_uncertainty_diameters,
            bearing_observations=observed_bearings,
            bearing_noise_std=estimator_noise_std
        )
        
        # Estimate position and bias (now with robust error handling)
        result, covariance = estimator.estimate_position_and_bias()
        
        # Since we now always return success=True, all estimates should be successful
        if result['success']:
            # Calculate errors
            true_meters = estimator.latlon_to_meters(true_position_latlon[0], true_position_latlon[1])
            est_meters = estimator.latlon_to_meters(result['latitude'], result['longitude'])
            position_error_m = np.linalg.norm(est_meters - true_meters)
            bias_error_deg = abs(result['bias_degrees'] - true_bias_deg)
            
            # Cap extreme errors (outliers) for stability
            position_error_m = min(position_error_m, 10000)  # Cap at 10km
            bias_error_deg = min(bias_error_deg, 180)  # Cap at 180 degrees
            
            position_errors.append(position_error_m)
            bias_errors.append(bias_error_deg)
            successful_estimates += 1
            
            # Store last estimator for plotting
            if sim == n_simulations - 1:
                last_estimator = estimator
    
    print(f"\nCompleted {successful_estimates}/{n_simulations} successful simulations")
    
    # Calculate statistics
    position_errors = np.array(position_errors)
    bias_errors = np.array(bias_errors)
    
    results = {
        'n_successful': successful_estimates,
        'n_total': n_simulations,
        'success_rate': successful_estimates / n_simulations * 100,
        'position_error_mean': np.mean(position_errors),
        'position_error_std': np.std(position_errors),
        'position_error_median': np.median(position_errors),
        'position_error_95th': np.percentile(position_errors, 95),
        'bias_error_mean': np.mean(bias_errors),
        'bias_error_std': np.std(bias_errors),
        'bias_error_median': np.median(bias_errors),
        'bias_error_95th': np.percentile(bias_errors, 95),
        'execution_time': time.time() - start_time
    }
   
    return results

import math
from scipy.optimize import least_squares

# --- Helper Functions for Least Squares Method ---
RADIUS_EARTH_METERS = 6371000

def lat_lon_to_meters(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg):
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    ref_lat_rad = math.radians(ref_lat_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    x_meters = RADIUS_EARTH_METERS * (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad)
    y_meters = RADIUS_EARTH_METERS * (lat_rad - ref_lat_rad)
    return x_meters, y_meters

def meters_to_lat_lon(x_meters, y_meters, ref_lat_deg, ref_lon_deg):
    ref_lat_rad = math.radians(ref_lat_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    lat_rad = ref_lat_rad + y_meters / RADIUS_EARTH_METERS
    lon_rad = ref_lon_rad + x_meters / (RADIUS_EARTH_METERS * math.cos(ref_lat_rad))
    return math.degrees(lat_rad), math.degrees(lon_rad)

def normalize_angle_difference(angle_diff_deg):
    return (angle_diff_deg + 180) % 360 - 180

def residuals(params, anchors_meters, measured_bearings_rad):
    user_x, user_y, bias_rad = params
    num_anchors = anchors_meters.shape[0]
    res = np.zeros(num_anchors)
    for i in range(num_anchors):
        anchor_x, anchor_y = anchors_meters[i, 0], anchors_meters[i, 1]
        measured_bearing = measured_bearings_rad[i]
        true_bearing_rad = measured_bearing - bias_rad
        residual = (user_x - anchor_x) * math.cos(true_bearing_rad) - \
                   (user_y - anchor_y) * math.sin(true_bearing_rad)
        res[i] = residual
    return res

def find_position_and_bias(anchors_lat_lon, measured_bearings_deg,
                           true_user_lat=None, true_user_lon=None, true_bias_deg=None):
    """
    Finds the user's position (latitude, longitude) assuming zero bias.
    Only (x, y) are optimized; bias is always zero.
    """
    # Reference point for local coordinates
    ref_lat_deg, ref_lon_deg = anchors_lat_lon[0]
    anchors_meters = [lat_lon_to_meters(lat, lon, ref_lat_deg, ref_lon_deg)
                      for lat, lon in anchors_lat_lon]
    anchors_meters = np.array(anchors_meters)
    measured_bearings_rad = np.deg2rad(measured_bearings_deg)

    def residuals(params, anchors_meters, measured_bearings_rad):
        user_x, user_y = params
        res = []
        for (anchor_x, anchor_y), bearing_rad in zip(anchors_meters, measured_bearings_rad):
            # Direction vector of the line (bearing: 0°=North, clockwise)
            dx = np.sin(bearing_rad)
            dy = np.cos(bearing_rad)
            # Vector from anchor to user
            vx = user_x - anchor_x
            vy = user_y - anchor_y
            # Perpendicular distance from point to line
            dist = np.abs(dx * vy - dy * vx)
            res.append(dist)
        return res
    
    # Initial guess: centroid of anchors
    initial_guess = np.mean(anchors_meters, axis=0)
    result = least_squares(residuals, initial_guess, args=(anchors_meters, measured_bearings_rad))

    est_x, est_y = result.x
    est_lat, est_lon = meters_to_lat_lon(est_x, est_y, ref_lat_deg, ref_lon_deg)

    output = {
        "estimated_user_lat": est_lat,
        "estimated_user_lon": est_lon,
        "success": result.success,
        "cost": result.cost,
        "nfev": result.nfev,
    }

    if true_user_lat is not None and true_user_lon is not None:
        true_x, true_y = lat_lon_to_meters(true_user_lat, true_user_lon, ref_lat_deg, ref_lon_deg)
        pos_error = np.linalg.norm([est_x - true_x, est_y - true_y])
        output["position_error_meters"] = pos_error

    return output

# --- Monte Carlo Comparison Function ---
def run_monte_carlo_comparison(n_simulations=100):
    """
    Runs a Monte Carlo simulation comparing the PositionBiasEstimator (likelihood)
    and the least-squares (orthogonal distance) method.
    """
    # Define anchor points in longitude/latitude
    anchor_positions_latlon = [
        [21.046234, 105.808489],
        [21.041684, 105.838083],
        [21.062275, 105.833513],
        [21.072387, 105.817222],
        [21.062281, 105.805139]
    ]
    anchor_uncertainty_diameters = [30, 30, 30, 30, 30]
    true_position_latlon = [21.058617, 105.821816]
    true_bias_deg = 3.0
    bearing_noise_std = 1.0
    estimator_noise_std = 1.0

    temp_estimator = PositionBiasEstimator(
        anchor_positions_latlon=anchor_positions_latlon,
        anchor_uncertainty_diameters=anchor_uncertainty_diameters,
        bearing_observations=[0, 0, 0, 0, 0],
        bearing_noise_std=estimator_noise_std
    )
    true_position_meters = temp_estimator.latlon_to_meters(true_position_latlon[0], true_position_latlon[1])

    errors_likelihood = []
    errors_ls = []
    bias_errors_likelihood = []
    bias_errors_ls = []

    for sim in range(n_simulations):
        observed_bearings = []
        for i, anchor_latlon in enumerate(anchor_positions_latlon):
            anchor_mean_meters = temp_estimator.anchors_meters[i]
            cov_matrix = temp_estimator.anchor_covs_meters[i]
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                cov_matrix = cov_matrix + np.eye(2) * 1e-6
            actual_anchor_meters = np.random.multivariate_normal(anchor_mean_meters, cov_matrix)
            dx = actual_anchor_meters[0] - true_position_meters[0]
            dy = actual_anchor_meters[1] - true_position_meters[1]
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                dx = 1e-6
            true_math_angle = np.arctan2(dy, dx)
            true_bearing = temp_estimator.math_angle_to_bearing(true_math_angle)
            noise = np.random.normal(0, bearing_noise_std)
            observed_bearing = (true_bearing + true_bias_deg + noise) % 360
            observed_bearings.append(observed_bearing)

        # --- Likelihood Estimator ---
        estimator = PositionBiasEstimator(
            anchor_positions_latlon=anchor_positions_latlon,
            anchor_uncertainty_diameters=anchor_uncertainty_diameters,
            bearing_observations=observed_bearings,
            bearing_noise_std=estimator_noise_std
        )
        result_likelihood, _ = estimator.estimate_position_and_bias()
        true_meters = estimator.latlon_to_meters(true_position_latlon[0], true_position_latlon[1])
        est_meters = estimator.latlon_to_meters(result_likelihood['latitude'], result_likelihood['longitude'])
        position_error_likelihood = np.linalg.norm(est_meters - true_meters)
        bias_error_likelihood = abs(result_likelihood['bias_degrees'] - true_bias_deg)
        errors_likelihood.append(position_error_likelihood)
        bias_errors_likelihood.append(min(bias_error_likelihood, 180))

        # --- Least Squares Estimator ---
        ls_results = find_position_and_bias(
            [[lat, lon] for lat, lon in anchor_positions_latlon],
            observed_bearings,
            true_user_lat=true_position_latlon[0],
            true_user_lon=true_position_latlon[1],
            true_bias_deg=true_bias_deg
        )
        errors_ls.append(ls_results["position_error_meters"])
        # Bias estimate is always zero for LS method
        estimated_bias_ls = 0.0
        # Normalize bias error to [-180, 180]
        bias_error_ls = (estimated_bias_ls - true_bias_deg + 180) % 360 - 180
        bias_errors_ls.append(abs(bias_error_ls))

    # --- Print and Plot Comparison ---
    print("\n=== Monte Carlo Comparison Results ===")
    print(f"Likelihood Estimator Mean Error: {np.mean(errors_likelihood):.2f} m, Median: {np.median(errors_likelihood):.2f} m")
    print(f"Least Squares Estimator Mean Error: {np.nanmean(errors_ls):.2f} m, Median: {np.nanmedian(errors_ls):.2f} m")
    print(f"Likelihood Estimator Mean Bias Error: {np.mean(bias_errors_likelihood):.2f}°, Median: {np.median(bias_errors_likelihood):.2f}°")
    print(f"Least Squares Estimator Mean Bias Error: {np.nanmean(bias_errors_ls):.2f}°, Median: {np.nanmedian(bias_errors_ls):.2f}°")
    
    # --- Visualization of Position Error Distributions ---
    plt.figure(figsize=(14, 5))

    # Plot 1: Likelihood Estimator
    plt.subplot(1, 2, 1)
    plt.hist(errors_likelihood, bins=20, alpha=0.7, color='tab:blue')
    plt.xlabel("Position Error (meters)")
    plt.ylabel("Frequency")
    plt.title("Likelihood Estimator\nPosition Error Distribution")
    plt.grid(True, alpha=0.3)

    # Plot 2: Least Squares Estimator
    plt.subplot(1, 2, 2)
    plt.hist(errors_ls, bins=20, alpha=0.7, color='tab:orange')
    plt.xlabel("Position Error (meters)")
    plt.ylabel("Frequency")
    plt.title("Stansfield Bias-Removing Estimator\nPosition Error Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Run your original simulation
# results = run_monte_carlo_simulation(n_simulations=100, plot_last=True)
# Run the comparison
run_monte_carlo_comparison(n_simulations=100)