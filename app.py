import streamlit as st
import numpy as np
from location_estimator import PositionBiasEstimator, find_position_and_bias

st.title("Location Estimation Demo")

st.markdown("""
Enter anchor positions and observed bearings below.  
You can also enter the ground truth location (optional) to see the error.
""")

# Dynamic number of anchors
n_anchors = st.number_input("Number of anchors", min_value=2, max_value=20, value=3, step=1)

anchor_positions_latlon = []
anchor_uncertainty_diameters = []
observed_bearings = []

st.subheader("Anchor Data")
for i in range(n_anchors):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lat = st.number_input(f"Anchor {i+1} Latitude", key=f"lat_{i}", value=21.04 + 0.01*i, format="%f")
    with col2:
        lon = st.number_input(f"Anchor {i+1} Longitude", key=f"lon_{i}", value=105.8 + 0.01*i, format="%f")
    with col3:
        bearing = st.number_input(f"Observed Bearing (deg)", key=f"bearing_{i}", value=45.0 + 10*i, format="%f")
    with col4:
        diameter = st.number_input(f"Uncertainty Diameter (m)", key=f"diam_{i}", value=30.0, min_value=0.0, format="%f")
    anchor_positions_latlon.append([lat, lon])
    anchor_uncertainty_diameters.append(diameter)
    observed_bearings.append(bearing)

st.subheader("Measurement Noise")
bearing_noise_std = st.number_input("Bearing noise std (deg)", min_value=0.0, value=7.0, format="%f")

st.subheader("Ground Truth (optional)")
col1, col2 = st.columns(2)
with col1:
    gt_lat = st.number_input("Ground Truth Latitude", value=0.0, format="%f")
with col2:
    gt_lon = st.number_input("Ground Truth Longitude", value=0.0, format="%f")
use_gt = st.checkbox("Use ground truth for error calculation", value=False)

if st.button("Estimate Location"):
    # --- Likelihood Estimator ---
    estimator = PositionBiasEstimator(
        anchor_positions_latlon=anchor_positions_latlon,
        anchor_uncertainty_diameters=anchor_uncertainty_diameters,
        bearing_observations=observed_bearings,
        bearing_noise_std=bearing_noise_std
    )
    result_likelihood, _ = estimator.estimate_position_and_bias()
    st.markdown("### Likelihood Estimator Result")
    st.write(f"Estimated Latitude: {result_likelihood['latitude']:f}")
    st.write(f"Estimated Longitude: {result_likelihood['longitude']:f}")
    st.write(f"Estimated Bias (deg): {result_likelihood['bias_degrees']:f}")

    # --- Least Squares Estimator ---
    ls_results = find_position_and_bias(
        anchor_positions_latlon,
        observed_bearings,
        true_user_lat=gt_lat if use_gt else None,
        true_user_lon=gt_lon if use_gt else None
    )
    st.markdown("### Least Squares Estimator Result")
    st.write(f"Estimated Latitude: {ls_results['estimated_user_lat']:f}")
    st.write(f"Estimated Longitude: {ls_results['estimated_user_lon']:f}")
    st.write(f"Estimated Bias (deg): 0.00 (assumed zero)")

    # --- Error Calculation ---
    if use_gt:
        st.markdown("### Positional Error")
        true_meters = estimator.latlon_to_meters(gt_lat, gt_lon)
        est_meters_lik = estimator.latlon_to_meters(result_likelihood['latitude'], result_likelihood['longitude'])
        error_lik = np.linalg.norm(est_meters_lik - true_meters)
        st.write(f"Likelihood Estimator Position Error: {error_lik:f} meters")

        est_meters_ls = estimator.latlon_to_meters(ls_results['estimated_user_lat'], ls_results['estimated_user_lon'])
        error_ls = np.linalg.norm(est_meters_ls - true_meters)
        st.write(f"Least Squares Estimator Position Error: {error_ls:f} meters")