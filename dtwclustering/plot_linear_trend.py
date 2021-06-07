import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import pygmt


def compute_interpolation(df):
    coordinates0 = np.column_stack((df['lon'].values, df['lat'].values))
    lonmin, lonmax = 120., 122.
    latmin, latmax = 21.8, 25.6
    step = 0.01
    lons = np.arange(lonmin, lonmax, step)
    lats = np.arange(latmin, latmax, step)

    xintrp, yintrp = np.meshgrid(lons, lats)
    z1 = griddata(coordinates0, df['slope'].values,
                  (xintrp, yintrp), method='nearest')
    xintrp = np.array(xintrp, dtype=np.float32)
    yintrp = np.array(yintrp, dtype=np.float32)

    z2 = z1[~np.isnan(z1)]

    cmapExtreme = np.max([np.abs(z2.min()), np.abs(z2.max())])

    da = xr.DataArray(z1, dims=("lat", "long"), coords={
                      "long": lons, "lat": lats},)

    return da, cmapExtreme


def plot_linear_trend_on_map(df, outfig="Maps/slope-plot.png"):
    da, cmapExtreme = compute_interpolation(df)

    minlon, maxlon = 120., 122.1
    minlat, maxlat = 21.8, 25.6

    frame = ["a1f0.25", "WSen"]
    # Visualization
    fig = pygmt.Figure()

    pygmt.makecpt(
        cmap='jet',
        series=f'{-cmapExtreme}/{cmapExtreme}/0.01',
        #     series='0/5000/100',
        continuous=True
    )

    # #plot high res topography
    fig.grdimage(
        region=[minlon, maxlon, minlat, maxlat],
        grid=da,
        projection='M4i',
        interpolation='l'
    )

    # plot coastlines
    fig.coast(
        region=[minlon, maxlon, minlat, maxlat],
        shorelines=True,
        water="#add8e6",
        frame=frame,
        area_thresh=1000
    )

    pygmt.makecpt(
        cmap='jet',
        series=f'{-cmapExtreme}/{cmapExtreme}/0.01',
        #     series='0/5000/100',
        continuous=True
    )

    # plot data points
    fig.plot(
        x=df['lon'].values,
        y=df['lat'].values,
        style='i0.2i',
        color=df['slope'].values,
        cmap=True,
        pen='black',
    )

    # Plot colorbar
    # Default is horizontal colorbar
    fig.colorbar(
        frame='+l"Linear Trend (mm)"'
    )

    # save figure as pdf
    fig.savefig(f"{outfig}", crop=True, dpi=300)

    print(f"Figure saved at {outfig}")


if __name__ == '__main__':
    pass
