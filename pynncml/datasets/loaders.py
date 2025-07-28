# Base on the code from:https://github.com/OpenSenseAction/OPENSENSE_sandbox/blob/main/notebooks/opensense_data_downloader_and_transformer.py
import os
import urllib.request
import zipfile
from functools import partial

import pandas as pd
import xarray as xr

from pynncml.datasets.dataset import LinkDataset
from pynncml.datasets.gauge_data import PointSensor
from pynncml.datasets import PointSet
import numpy as np
from datetime import datetime


from pynncml.datasets.xarray_processing import xarray2link


def download_data_file(url, local_path=".", local_file_name=None, print_output=True):
    """
    Download a file from a URL to a local path
    :param url: URL to download from
    :param local_path: Local path to download to
    :param local_file_name: Local file name to save as
    :param print_output: Print download information
    """
    if not os.path.exists(local_path):
        if print_output:
            print(f"Creating path {local_path}")
        os.makedirs(local_path)

    if local_file_name is None:
        local_file_name = url.split("/")[-1]

    if os.path.exists(os.path.join(local_path, local_file_name)):
        print(
            f"File already exists at desired location {os.path.join(local_path, local_file_name)}"
        )
        print("Not downloading!")
        return

    if print_output:
        print(f"Downloading {url}")
        print(f"to {local_path}/{local_file_name}")

    request_return_meassage = urllib.request.urlretrieve(
        url, os.path.join(local_path, local_file_name)
    )
    return request_return_meassage


download_open_mrg = partial(
    download_data_file,
    url="https://zenodo.org/record/7107689/files/OpenMRG.zip",
)


def transform_open_mrg(fn, path_to_extract_to):
    """
    Transform the OpenMRG dataset to a xarray dataset
    :param fn: File name
    :param path_to_extract_to: Path to extract to
    """
    # For this ZIP file we cannot extract only the CML dataset since
    # the NetCDF with the CML dataset is quite large. This seems to
    # lead to crashes when reding directly from the ZIP file via Python.
    with zipfile.ZipFile(fn) as zfile:
        zfile.extractall(path_to_extract_to)

    # Read metadata and dataset
    df_metadata = pd.read_csv(os.path.join(path_to_extract_to, 'cml/cml_metadata.csv'), index_col=0)
    ds = xr.open_dataset(os.path.join(path_to_extract_to, 'cml/cml.nc'))

    # Add metadata with naming convention as currently used in pycomlink example dataset file
    for col_name, ds_var_name in [
        ('NearLatitude_DecDeg', 'site_0_lat'),
        ('NearLongitude_DecDeg', 'site_0_lon'),
        ('FarLatitude_DecDeg', 'site_1_lat'),
        ('FarLongitude_DecDeg', 'site_1_lon'),
        ('Frequency_GHz', 'frequency'),
        ('Polarization', 'polarization'),
        ('Length_km', 'length'),
    ]:
        ds.coords[ds_var_name] = (
            ('sublink'),
            [df_metadata[df_metadata.Sublink == sublink_id][col_name].values[0] for sublink_id in
             list(ds.sublink.values)]
        )

    ds.attrs['comment'] += '\nMetadata added with preliminary code from opensense_data_downloader.py'

    # Change "sublink" to "sublink_id"
    ds = ds.rename({"sublink": "sublink_id"})

    return ds


def rain2rain_rate(in_array: np.ndarray, window_size: int = 15, step_time: int = 60):
    """
    Convert rain to rain rate
    :param in_array: Input array
    :param window_size: Window size
    :param step_time: Step time
    """
    res = np.zeros(in_array.shape[0])
    scale = np.zeros(in_array.shape[0])
    start = False

    for i in reversed(range(in_array.shape[0])):
        if in_array[i] == 0.0:
            if start and (index - i) >= window_size:
                v = in_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / step_time)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
                start = False
        else:
            if start:
                v = in_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / step_time)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
            index = i
            start = True
    res = res * scale
    return np.convolve(res, np.ones(window_size) * (1 / window_size), mode='same')


def load_open_mrg(data_path="./data/",
                  restriction_minimum_length=0,
                  change2min_max=False,
                  xy_min=None,
                  xy_max=None,
                  time_slice=None,
                  rain_gauge_time_base=900,
                  link2gauge_distance=2000,
                  sampling_interval_in_sec = 10,
                  samples_type="min_max"):
    download_open_mrg(local_path=data_path)
    file_location = data_path + "OpenMRG.zip"
    ds = transform_open_mrg(file_location, data_path)

    if time_slice is not None:
        ds = ds.sel(time=time_slice)

    time_array = ds.time.to_numpy().astype('datetime64[s]')
    ###########################################
    # Process Gauge
    ###########################################
    gauge_metadata = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-metadata.csv'), index_col=0)
    gauge_data = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-2015JJA.csv'), index_col=0)
    time_array_gauge = np.asarray([np.datetime64(i[:-1]) for i in gauge_data.index.to_numpy()])
    sel_index = np.logical_and(time_array_gauge >= time_array[0], time_array_gauge <= time_array[-1])
    gauge_list = []
    for g_id in gauge_data.keys():
        gauge_data_array = gauge_data.get(g_id).values[sel_index]

        # ✅ Fix for missing first sample at 2015-06-01 00:00:00
        if str(time_slice.start) == "2015-06-01":
            gauge_data_array = np.insert(gauge_data_array, 0, gauge_data_array[0])  # duplicate first value

        rain_rate_gauge = rain2rain_rate(gauge_data_array, window_size = 15)
        i = np.where(gauge_metadata.index == g_id)[0]
        lon = gauge_metadata.get("Longitude_DecDeg").values[i]
        lat = gauge_metadata.get("Latitude_DecDeg").values[i]
        if not np.any(np.isnan(rain_rate_gauge)):
            ps = PointSensor(rain_rate_gauge, time_array_gauge.astype("int")[sel_index], lat, lon)
            ps = ps.change_time_base(rain_gauge_time_base)
            gauge_list.append(ps)
    ps = PointSet(gauge_list)
    ###########################################
    # Process Links
    ###########################################
    # Prints for debug:
    n_days = (datetime.fromisoformat(str(time_slice.stop)) - datetime.fromisoformat(str(time_slice.start))).days + 1
    print(f"\n📅 Time slice: {time_slice.start} to {time_slice.stop} → {n_days} days")
    expected_link_samples = int((24 * 60 * 60 * n_days) / sampling_interval_in_sec)
    print(f"Inspecting first link...")
    # End debug.
    link_set = xarray2link(ds,
                           link2gauge_distance,
                           ps,
                           xy_max,
                           xy_min,
                           restriction_minimum_length=restriction_minimum_length,
                           change2min_max=change2min_max,
                           samples_type=samples_type,
                           sampling_interval_in_sec = sampling_interval_in_sec)

    # Prints for debug:
    if link_set.link_list:
        link = link_set.link_list[0]
        timestamps = link.time_array
        n = len(timestamps)

        # Format timestamps
        first_four = [datetime.utcfromtimestamp(t).isoformat() for t in timestamps[:4]]
        last_four = [datetime.utcfromtimestamp(t).isoformat() for t in timestamps[-4:]]
        print(f"Downsample CML samples count: {n}")
        print(f"🕒 First 4 DOWNSAMPLED CML timestamps: {first_four}")
        print(f"🕓 Last 4 DOWNSAMPLED CML timestamps:  {last_four}")


        # Δt and duration check
        dt = np.diff(timestamps[:2])[0]
        if samples_type != "original":
            assert np.isclose(dt, sampling_interval_in_sec, atol=1), f"❌ Link Δt mismatch: expected {sampling_interval_in_sec}, got {dt}"
            print(f"✅ Time step Δt = {dt}s")
            assert n == expected_link_samples, f"❌ Sample count mismatch: expected {expected_link_samples}, got {n}"
            print(f"✅ Sample count = {expected_link_samples}")
        else:
            print(f"ℹ️ Original sampling Δt = {dt}s (no downsampling applied)")

        print(f"Type of link: {type(link)}")

        # Print sublink ID if available
        if hasattr(link, "sublink_id"):
            print(f"🆔 Sublink ID: {link.sublink_id}")

        print("Link Features (full attribute scan):")
        for key, value in link.__dict__.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")
    else:
        print("⚠️ No links loaded for this region/time.")
    # End debug.
    return link_set, ps


def loader_open_mrg_dataset(data_path="./data/",
                            restriction_minimum_length=0,
                            xy_min=None,
                            xy_max=None,
                            time_slice=None,
                            link2gauge_distance=2000,
                            samples_type="min_max",
                            sampling_interval_in_sec = 10):
    """
    Load OpenMRG dataset
    :param data_path: Path to store the dataset
    :param xy_min: Minimum xy use to filter the dataset based on xy location
    :param xy_max: Maximum xy use to filter the dataset based on xy location
    :param time_slice: Time slice to filter the dataset
    :param link2gauge_distance: Link to gauge distance in meter
    :param samples_type: "min_max"(default) or "instantaneous"
    :param restriction_minimum_length: Minimum link length in kilometers (default is 0 = no restriction)
    :param sampling_interval_in_sec: used only when samples_type == "instantaneous"
    :return: LinkDataset
    """
    # Set parameters based on sampling type
    if samples_type == "min_max":
        change2min_max = True
    else:
        change2min_max = False


    link_set, point_set = load_open_mrg(data_path = data_path,
                                        restriction_minimum_length=restriction_minimum_length,
                                        change2min_max = change2min_max,
                                        xy_min = xy_min,
                                        xy_max = xy_max,
                                        time_slice = time_slice,
                                        rain_gauge_time_base = 900,
                                        link2gauge_distance = link2gauge_distance,
                                        sampling_interval_in_sec = sampling_interval_in_sec,
                                        samples_type = samples_type)
    return LinkDataset(link_set, point_set)
