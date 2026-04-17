"""
Methods to interpolate AIS data every dt using csv files.

The conversion is done to emulate a sensor with constant sampling rate, such as camera.

Example of AIS data format:

,timestamp,type_of_mobile,mmsi,lat,lon,nav_status,rot,sog,cog,heading,imo,callsign,name,ship_type,cargo_type,length,width,type_pos_device,draught,dest,eta,data_source,a,b,c,d,distance_m,timestamp_sec
0,2023-04-06 04:55:04,Class A,265610940,56.0448,12.6879,Under way using engine,0.0,0.0,344.1,67.0,Unknown,SCDW,PILOT 211 SE,Pilot,No additional information,17.0,5.0,GPS,,DERSK,06/10/2023 06:00:00,AIS,10.0,7.0,3.0,2.0,271.4048252685546,1680756904.0
1,2023-04-06 04:55:04,Class A,219022265,56.0456,12.6868,Under way using engine,0.0,0.0,353.2,344.0,9788124,OXTC2,SVITZER HERMOD,Tug,No additional information,28.0,12.0,Combined GPS/GLONASS,5.5,SEHEL,05/04/2023 14:00:00,AIS,12.0,16.0,5.0,7.0,381.9655367523393,1680756904.0
2,2023-04-06 04:55:05,Class A,219014974,56.0452,12.6916,Moored,0.0,0.0,131.7,239.0,8010532,OYOB2,PERNILLE,Passenger,Reserved for future use,36.0,8.0,GPS,2.2,HBG<->HLO,14/05/2023 19:00:00,AIS,9.0,27.0,5.0,3.0,223.77318206655684,1680756905.0
3,2023-04-06 04:55:06,Class A,265041000,56.0432,12.6912,Under way using engine,0.0,0.0,63.9,70.0,9007128,SCQX,AURORA AF HELSINGBOR,Passenger,No additional information,111.0,28.0,GPS,5.0,HGB-HLO-HGB,25/04/2023 07:00:00,AIS,31.0,80.0,14.0,14.0,0.0,
4,2023-04-06 04:55:13,Class A,219000368,56.0441,12.6906,Under way using engine,0.0,0.0,336.0,244.0,8611685,OXIE2,MERCANDIA IV,Passenger,No additional information,96.0,15.0,GPS,3.5,HELSINGOR.HELSINGBOR,06/04/2023 07:00:00,AIS,46.0,50.0,8.0,7.0,106.78862276983217,1680756913.0
5,2023-04-06 04:55:13,Class A,219022265,56.0456,12.6868,Under way using engine,0.0,0.0,353.2,344.0,9788124,OXTC2,SVITZER HERMOD,Tug,No additional information,28.0,12.0,Combined GPS/GLONASS,5.5,SEHEL,05/04/2023 14:00:00,AIS,12.0,16.0,5.0,7.0,381.9655367523393,1680756913.0
6,2023-04-06 04:55:14,Class A,265504660,56.0448,12.6877,Under way using engine,,0.0,0.0,,Unknown,SFYA,PILOT 741 SE,Pilot,Reserved for future use,16.0,5.0,GPS,,SKAGEN,04/04/2023 20:00:00,AIS,12.0,4.0,4.0,1.0,280.90345312846216,1680756914.0
7,2023-04-06 04:55:15,Class A,265041000,56.0432,12.6912,Under way using engine,0.0,0.0,63.9,70.0,9007128,SCQX,AURORA AF HELSINGBOR,Passenger,Reserved for future use,111.0,28.0,GPS,5.0,HGB-HLO-HGB,17/02/2024 05:00:00,AIS,31.0,80.0,14.0,14.0,0.0,
8,2023-04-06 04:55:23,Class A,219022265,56.0456,12.6868,Under way using engine,0.0,0.0,353.2,344.0,9788124,OXTC2,SVITZER HERMOD,Tug,No additional information,28.0,12.0,Combined GPS/GLONASS,5.5,SEHEL,05/04/2023 14:00:00,AIS,12.0,16.0,5.0,7.0,381.9655367523393,1680756923.0
9,2023-04-06 04:55:24,Class A,219000368,56.0441,12.6906,Under way using engine,0.0,0.0,4.0,244.0,8611685,OXIE2,MERCANDIA IV,Passenger,No additional information,96.0,15.0,GPS,3.5,HELSINGOR.HELSINGBOR,13/09/2023 15:00:00,AIS,46.0,50.0,8.0,7.0,106.78862276983217,1680756924.0
...
...
...
798,2023-04-06 05:29:31,Class A,220252000,56.0376,12.616,Moored,,0.1,56.7,,Unknown,OWID2,ZAR,Undefined,No additional information,53.0,7.0,GPS,2.8,HELSING,01/11/2023 12:00:00,AIS,38.0,15.0,5.0,2.0,562.7361184396926,1680758971.0
799,2023-04-06 05:29:35,Class A,265041000,56.0326,12.6174,Under way using engine,0.0,0.2,264.1,77.0,9007128,SCQX,AURORA AF HELSINGBOR,Passenger,No additional information,111.0,28.0,GPS,5.0,HGB-HLO-HGB,31/12/2023 23:59:00,AIS,31.0,80.0,14.0,14.0,0.0,
800,2023-04-06 05:29:49,Class A,211226940,56.036,12.6167,Under way using engine,0.0,0.0,253.5,146.0,5104253,DMKD,SEHO,Passenger,Category X,25.0,8.0,GPS,,HEILIGENHAFEN,17/12/2023 17:00:00,AIS,8.0,17.0,6.0,2.0,380.5555916509611,1680758989.0
801,2023-04-06 05:29:59,Class A,211226940,56.036,12.6167,Under way using engine,0.0,0.0,253.5,146.0,5104253,DMKD,SEHO,Passenger,No additional information,25.0,8.0,GPS,,HEILIGENHAFEN,17/12/2023 17:00:00,AIS,8.0,17.0,6.0,2.0,380.5555916509611,1680758999.0

TODO: Create a method that loads a csv file inside a pd dataframe based on its filename
TODO: Create a method that interpolate the AIS data from a pd dataframe based on a specified target sampling rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def load_ais_csv(filename):
    """Load AIS CSV file into pandas DataFrame."""
    return pd.read_csv(filename)


def interpolate_ais_data(df, dt=1.0, smooth=True, sigma=2.0, exclude_ship=None) -> pd.DataFrame:
    """
    Interpolate AIS data to regular time intervals with optional smoothing.
    
    Parameters:
    - df: pandas DataFrame with AIS data
    - dt: time interval in seconds for interpolation
    - smooth: apply Gaussian smoothing to interpolated data
    - sigma: standard deviation for Gaussian smoothing (higher = smoother)
    - exclude_ship: MMSI number or ship name to exclude from interpolation
    
    Returns:
    - DataFrame with interpolated and optionally smoothed data at regular intervals
    """
    # Use timestamp_sec for interpolation
    df = df.copy()
    df['timestamp_sec'] = pd.to_numeric(df['timestamp_sec'], errors='coerce')
    df = df.dropna(subset=['timestamp_sec'])
    
    # Exclude specified ship if requested
    if exclude_ship is not None:
        # Check if exclude_ship matches MMSI or name
        mmsi_mask = df['mmsi'] != exclude_ship
        name_mask = True
        if 'name' in df.columns:
            name_mask = df['name'] != exclude_ship
        df = df[mmsi_mask & name_mask]
    
    # Group by MMSI (vessel ID) and interpolate each vessel separately
    interpolated_data = []
    
    for mmsi in df['mmsi'].unique():
        vessel_data = df[df['mmsi'] == mmsi].copy()
        vessel_data = vessel_data.sort_values('timestamp_sec')
        
        # Create time grid for THIS vessel only (not global range)
        vessel_start_time = vessel_data['timestamp_sec'].min()
        vessel_end_time = vessel_data['timestamp_sec'].max()
        new_times = np.arange(vessel_start_time, vessel_end_time + dt, dt)
        
        # Interpolate numeric columns
        numeric_cols = ['lat', 'lon', 'sog', 'cog', 'heading']
        
        # Create new DataFrame for this vessel
        vessel_interp = pd.DataFrame({'timestamp_sec': new_times})
        vessel_interp['mmsi'] = mmsi
        
        # Copy non-numeric fields from first record
        for col in ['name', 'ship_type', 'length', 'width']:
            if col in vessel_data.columns:
                vessel_interp[col] = vessel_data[col].iloc[0]
        
        # Interpolate numeric fields
        for col in numeric_cols:
            if col in vessel_data.columns:
                # Basic linear interpolation
                interpolated_values = np.interp(
                    new_times, 
                    vessel_data['timestamp_sec'], 
                    vessel_data[col].fillna(method='ffill')
                )
                
                # Apply smoothing if requested
                if smooth and len(interpolated_values) > 3:
                    # Handle circular angles (cog, heading) differently
                    if col in ['cog', 'heading']:
                        # Convert to unit vectors, smooth, then back to angles
                        angles_rad = np.deg2rad(interpolated_values)
                        cos_vals = ndimage.gaussian_filter1d(np.cos(angles_rad), sigma=sigma)
                        sin_vals = ndimage.gaussian_filter1d(np.sin(angles_rad), sigma=sigma)
                        smoothed_angles = np.rad2deg(np.arctan2(sin_vals, cos_vals))
                        # Ensure positive angles
                        smoothed_angles = (smoothed_angles + 360) % 360
                        vessel_interp[col] = smoothed_angles
                    else:
                        # Regular smoothing for lat, lon, sog
                        vessel_interp[col] = ndimage.gaussian_filter1d(interpolated_values, sigma=sigma)
                else:
                    vessel_interp[col] = interpolated_values
            else:
                vessel_interp[col] = np.nan
        
        interpolated_data.append(vessel_interp)
    
    return pd.concat(interpolated_data, ignore_index=True)


def plot_ship_trajectories(ax, csv_filename, dt_psi=None):
    """
    Plot trajectories of all ships from CSV file on given matplotlib axes.
    
    Parameters:
    - ax: matplotlib Axes object
    - csv_filename: path to AIS CSV file
    - dt_psi: time interval in seconds for plotting heading arrows (None = no arrows)
    """
    # Load data
    df = load_ais_csv(csv_filename)
    
    # Convert coordinates to numeric
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['timestamp_sec'] = pd.to_numeric(df['timestamp_sec'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    
    # Plot each ship's trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['mmsi'].unique())))
    
    for i, mmsi in enumerate(df['mmsi'].unique()):
        ship_data = df[df['mmsi'] == mmsi].copy()
        ship_data = ship_data.sort_values('timestamp_sec')
        
        # Get ship name for legend
        ship_name = ship_data['name'].iloc[0] if 'name' in ship_data.columns else f'MMSI {mmsi}'
        
        # Plot trajectory (lon=east=x, lat=north=y)
        ax.plot(ship_data['lon'], ship_data['lat'], 
                color=colors[i], linewidth=2, label=ship_name)
        
        # Plot heading arrows if requested
        if dt_psi is not None and 'heading' in ship_data.columns and not ship_data['heading'].isna().all():
            # Select points at dt_psi intervals
            start_time = ship_data['timestamp_sec'].min()
            end_time = ship_data['timestamp_sec'].max()
            arrow_times = np.arange(start_time, end_time, dt_psi)
            
            for arrow_time in arrow_times:
                # Find closest time point
                closest_idx = (ship_data['timestamp_sec'] - arrow_time).abs().idxmin()
                row = ship_data.loc[closest_idx]
                
                if not pd.isna(row['heading']):
                    # Convert heading to direction vector
                    # Heading: 0°=North, 90°=East (nautical convention)
                    # Convert to math convention: 0°=East, 90°=North
                    heading_math = 90 - row['heading']
                    dx = np.cos(np.deg2rad(heading_math))
                    dy = np.sin(np.deg2rad(heading_math))
                    
                    # Scale arrow size (adjust as needed)
                    scale = 0.002
                    ax.arrow(row['lon'], row['lat'], 
                            dx * scale, dy * scale,
                            head_width=scale*0.5, head_length=scale*0.5,
                            fc=colors[i], ec=colors[i], alpha=0.7)
    
    ax.set_xlabel('Longitude (East)')
    ax.set_ylabel('Latitude (North)')
    ax.set_title('Ship Trajectories')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    import os, glob
    os.makedirs(os.path.join('data', 'smooth_interp'), exist_ok=True)
    files = glob.glob(os.path.join('data', 'raw', '*.csv'))

    for f in files:
        df = load_ais_csv(f)
        df_smooth_interp = interpolate_ais_data(df, dt=1.0, smooth=True, sigma=10.0, exclude_ship=265041000)
        df_smooth_interp.to_csv(os.path.join('data', 'smooth_interp', os.path.basename(f)))

    # test_filename = os.path.join('data', 'AIS2.csv')
    
    # # Test loading
    # loaded_df = load_ais_csv(test_filename)
    # print("Loaded AIS data:")
    # print(loaded_df.head())

    # # Test interpolation with smoothing
    # interpolated_df_smooth = interpolate_ais_data(loaded_df, dt=1.0, smooth=True, sigma=10.0, exclude_ship=265041000)
    # print("Interpolated AIS data with smoothing (1 second intervals):")
    # print(interpolated_df_smooth.head())

    # interpolated_df_smooth.to_csv(os.path.join('data', 'AIS_interp_smooth.csv'))
    
    # Compare trajectories with and without smoothing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    test_file = files[0]


    plot_ship_trajectories(ax1, test_file)
    ax1.set_title('Original AIS Data')
    
    # Create a temporary file for smoothed data to use with plot function
    # temp_smooth_file = os.path.join('data', 'AIS_interp_smooth.csv')
    
    # Plot with heading arrows every 60 seconds
    plot_ship_trajectories(ax2, os.path.join('data', 'smooth_interp', os.path.basename(test_file)))#, dt_psi=60)
    ax2.set_title('Smoothed Data with Heading Arrows')
    
    plt.tight_layout()
    plt.show()
    