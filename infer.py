from location_estimator import PositionBiasEstimator, find_position_and_bias
import numpy as np

# --- EDIT THESE VARIABLES ---
anchor_positions_latlon = [
    [21.03216283561491, 105.81280185106479],
    [21.044917207324673, 105.84352963832035],
    [21.059682869604966, 105.8316663239166]
]
anchor_uncertainty_diameters = [97, 40, 64]
observed_bearings = [198.67, 109.67, 60]
bearing_noise_std = 5.0

# Set to None if you don't want error calculation
ground_truth = {
    "latitude": 21.054144,
    "longitude": 105.820591
}
# ground_truth = None  # Uncomment this line to skip error calculation

# --- END OF EDITABLE SECTION ---

def main():
    print("=== Location Estimation CLI ===")
    print(f"Number of anchors: {len(anchor_positions_latlon)}")

    # --- Likelihood Estimator ---
    estimator = PositionBiasEstimator(
        anchor_positions_latlon=anchor_positions_latlon,
        anchor_uncertainty_diameters=anchor_uncertainty_diameters,
        bearing_observations=observed_bearings,
        bearing_noise_std=bearing_noise_std
    )
    result_likelihood, _ = estimator.estimate_position_and_bias()
    print("\n=== Likelihood Estimator Result ===")
    print(f"Estimated Latitude: {result_likelihood['latitude']:.8f}")
    print(f"Estimated Longitude: {result_likelihood['longitude']:.8f}")
    print(f"Estimated Bias (deg): {result_likelihood['bias_degrees']:.4f}")

    # --- Least Squares Estimator ---
    ls_results = find_position_and_bias(
        anchor_positions_latlon,
        observed_bearings,
        true_user_lat=ground_truth["latitude"] if ground_truth else None,
        true_user_lon=ground_truth["longitude"] if ground_truth else None
    )
    print("\n=== Least Squares Estimator Result ===")
    print(f"Estimated Latitude: {ls_results['estimated_user_lat']:.8f}")
    print(f"Estimated Longitude: {ls_results['estimated_user_lon']:.8f}")
    print(f"Estimated Bias (deg): 0.00 (assumed zero)")

    # --- Error Calculation ---
    if ground_truth:
        print("\n=== Positional Error ===")
        true_meters = estimator.latlon_to_meters(ground_truth["latitude"], ground_truth["longitude"])
        est_meters_lik = estimator.latlon_to_meters(result_likelihood['latitude'], result_likelihood['longitude'])
        error_lik = np.linalg.norm(est_meters_lik - true_meters)
        print(f"Likelihood Estimator Position Error: {error_lik:.4f} meters")

        est_meters_ls = estimator.latlon_to_meters(ls_results['estimated_user_lat'], ls_results['estimated_user_lon'])
        error_ls = np.linalg.norm(est_meters_ls - true_meters)
        print(f"Least Squares Estimator Position Error: {error_ls:.4f} meters")

if __name__ == "__main__":
    main()