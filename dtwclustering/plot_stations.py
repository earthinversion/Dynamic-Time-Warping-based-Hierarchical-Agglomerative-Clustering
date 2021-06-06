import numpy as np
import pygmt
import pandas as pd
import os
np.random.seed(45)  # to get the same color at each run


def plot_station_map(station_data, minlon=None, maxlon=None, minlat=None, maxlat=None, outloc="Maps"):
    os.makedirs(outloc, exist_ok=True)
    df = pd.read_csv(station_data)
    print(df.head())

    if minlon is None and maxlon is None and minlat is None and maxlat is None:
        minlon = df['lon'].min()-1
        maxlon = df['lon'].max()+1
        minlat = df['lat'].min()-1
        maxlat = df['lat'].max()+1

    colorsList = ['blue']

    # define etopo data file
    topo_data = "@earth_relief_15s"

    # Visualization
    fig = pygmt.Figure()
    # make color pallets
    pygmt.makecpt(
        cmap='etopo1',
        series='-8000/5000/1000',
        continuous=True
    )

    # plot high res topography
    fig.grdimage(
        grid=topo_data,
        region=[minlon, maxlon, minlat, maxlat],
        projection='M4i',
        shading=True,
        frame=True
    )

    # plot coastlines
    fig.coast(
        region=[minlon, maxlon, minlat, maxlat],
        projection='M4i',
        shorelines=True,
        frame=True
    )
    leftjustify, rightoffset = "TL", "5p/-5p"
    fig.plot(
        x=df["lon"].values,
        y=df["lat"].values,
        style="i10p",
        color=colorsList[0],
        pen="black",
        label="Stations"
    )

    for snum in range(df.shape[0]):
        fig.text(
            x=df.loc[snum, 'lon'],
            y=df.loc[snum, 'lat'],
            text=f"{df.loc[snum, 'stn']}",
            justify=leftjustify,
            angle=0,
            offset=rightoffset,
            fill="white",
            font=f"6p,Helvetica-Bold,black",
        )

    fig.legend(position="JTR+jTR+o0.2c", box=True)

    fig.savefig(os.path.join(outloc, 'station_map.png'), crop=True, dpi=300)
    print(f"Output figure saved at {os.path.join(outloc, 'station_map.png')}")


if __name__ == '__main__':
    plot_station_map()
