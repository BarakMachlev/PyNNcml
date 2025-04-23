import numpy as np
from tqdm import tqdm

from pynncml.datasets import MetaData, Link, LinkSet


def xarray_time_slice(ds, start_time, end_time):
    """
    Slice the xarray dataset based on time
    :param ds: xarray dataset
    :param start_time: start time
    :param end_time: end time
    :return: xarray dataset
    """
    return ds.sel(time=slice(start_time, end_time))


def xarray_location_slice(ds, lon_min, lon_max, lat_min, lat_max):
    """
    Slice the xarray dataset based on location
    :param ds: xarray dataset
    :param lon_min: min longitude
    :param lon_max: max longitude
    :param lat_min: min latitude
    :param lat_max: max latitude
    """
    return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def xarray_sublink2link(ds_sublink, gauge=None):
    """
    Convert xarray sublink to link
    :param ds_sublink: xarray dataset
    :param gauge: gauge data
    :return: Link
    """
    md = MetaData(float(ds_sublink.frequency),
                  "Vertical" in str(ds_sublink.polarization),
                  float(ds_sublink.length),
                  None,
                  None,
                  lon_lat_site_zero=[float(ds_sublink.site_0_lon), float(ds_sublink.site_0_lat)],
                  lon_lat_site_one=[float(ds_sublink.site_1_lon), float(ds_sublink.site_1_lat)])
    rsl = ds_sublink.rsl.to_numpy()
    tsl = ds_sublink.tsl.to_numpy()

    # Prints for debug:
    time_array = ds_sublink.time.to_numpy().astype('datetime64[s]').astype("int")
    if not hasattr(xarray_sublink2link, "_printed"):  # Only print for first link
        from datetime import datetime
        first_4 = [datetime.utcfromtimestamp(t).isoformat() for t in time_array[:4]]
        last_4 = [datetime.utcfromtimestamp(t).isoformat() for t in time_array[-4:]]
        print()
        print(f"Original CML samples count: {len(time_array)}")
        duration_sec = time_array[-1] - time_array[0]
        print(f"Total raw duration: {duration_sec} seconds ({duration_sec / 3600:.2f} hours)")
        print(f"🕒 First 4 ORIGINAL CML timestamps: {first_4}")
        print(f"🕓 Last 4 ORIGINAL CML timestamps:  {last_4}")
        xarray_sublink2link._printed = True
    # End debug.

    if np.any(np.isnan(rsl)):
        for nan_index in np.where(np.isnan(rsl))[0]:
            rsl[nan_index] = rsl[nan_index - 1]
    if np.any(np.isnan(tsl)):
        tsl[np.isnan(tsl)] = np.unique(tsl)[0]
    if not np.any(np.isnan(rsl)) and not np.any(np.isnan(tsl)):
        link = Link(rsl,
                    ds_sublink.time.to_numpy().astype('datetime64[s]').astype("int"),
                    meta_data=md,
                    rain_gauge=None,
                    link_tsl=tsl,
                    gauge_ref=gauge)
    else:
        link = None
    return link


def xarray2link(ds,
                link2gauge_distance,
                ps,
                xy_max=None,
                xy_min=None,
                change2min_max=False,
                samples_type="min_max",
                window_size_in_sec: int = 900):
    """
    Convert xarray dataset to a LinkSet object.

    :param ds: xarray dataset loaded from NetCDF
    :param link2gauge_distance: max allowed distance (in meters) to associate a link with a gauge
    :param ps: PointSet containing rain gauge sensors
    :param xy_max: upper spatial bound [x, y]
    :param xy_min: lower spatial bound [x, y]
    :param change2min_max: whether to apply min/max compression (used only if samples_type == "min_max")
    :param samples_type: "min_max" or "instantaneous"
    :param window_size_in_sec: window size in seconds (e.g., 900 = 15 min, 60 = 1 min)
    :return: LinkSet object with filtered and compressed links
    """

    link_list = []
    for i in tqdm(range(len(ds.sublink_id))):
        ds_sublink = ds.isel(sublink_id=i)
        md = MetaData(float(ds_sublink.frequency),
                      "Vertical" in str(ds_sublink.polarization),
                      float(ds_sublink.length),
                      None,
                      None,
                      lon_lat_site_zero=[float(ds_sublink.site_0_lon), float(ds_sublink.site_0_lat)],
                      lon_lat_site_one=[float(ds_sublink.site_1_lon), float(ds_sublink.site_1_lat)])
        xy_array = md.xy()
        if xy_min is None or xy_max is None:
            x_check = y_check = True
        else:
            x_check = xy_min[0] < xy_array[0] and xy_min[0] < xy_array[2] and xy_max[0] > xy_array[2] and xy_max[0] > \
                      xy_array[0]

            y_check = xy_min[1] < xy_array[1] and xy_min[1] < xy_array[3] and xy_max[1] > xy_array[3] and xy_max[1] > \
                      xy_array[1]

        if x_check and y_check:
            if ps == None:
                link = xarray_sublink2link(ds_sublink)
            else:
                d_min, gauge = ps.find_near_gauge(md.xy_center())
                if d_min < link2gauge_distance:
                    link = xarray_sublink2link(ds_sublink, gauge)
                else:
                    link = None  # Link is too far from the gauge
            if link is not None:
                if samples_type == "min_max":
                    link = link.create_min_max_link(window_size_in_sec)
                elif samples_type == "instantaneous":
                    link = link.create_compressed_instantaneous_link(window_size_in_sec)
                elif samples_type == "original":
                    pass  # Leave the link as-is (no compression or resampling)

                link_list.append(link)

    return LinkSet(link_list)
