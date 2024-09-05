import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MPLPath

# Overrides openMDAO Problem.check_config to suppress output to a file.
from openmdao.core.problem import Problem
original_check_config = Problem.check_config
def check_config_alt(*args, **kwargs):
    """
    Overrides openMDAO Problem.check_config to suppress output to a file.
    """
    return original_check_config(*args, out_file=None, **kwargs)
Problem.check_config = check_config_alt

from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import V80
from py_wake.examples.data.iea37 import IEA37_WindTurbines
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.site import XRSite
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.gradients import autograd
from py_wake.utils.plotting import setup_plot
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from shapely.geometry import Polygon as ShapelyPolygon
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp
import viktor as vkt

from constants import IMAGE_DPI, ROOT, get_divisors, serialize
from gwa_reader import get_gwc_data  # Global Wind Atlas API

# wind turbines
TURBINES = ["V80 (2.0 MW)", "IEA37 (3.35 MW)", "DTU (10 MW)"]
TURBINE_CLASSES = [V80, IEA37_WindTurbines, DTU10MW]
TURBINE_CLASSES_DICT = {
    turbine: turbine_cls for turbine, turbine_cls in zip(TURBINES, TURBINE_CLASSES)
}

# physical model
ROUGHNESS_INDEX = 0  # we assume a flat ground (reasonable for off-shore farms)
WIND_MEASUREMENT_HEIGHT = 100.0  # m (for convenience we fix this height. Not accurate as turbine hub heights may vary)
TURBULENCE_INTENSITY = 0.1
WIND_BIN_NUMS = get_divisors(360)
SITE_MINIMUM_AREA = 1  # km^2
SITE_MAXIMUM_AREA = 20  # km^2
OPENMDAO_OUT_PATH = ROOT / "openmdao_checks.out"

# optimization
MAX_ITERATIONS = 20


def get_windfarm_area(params, **kwargs):
    """
    Calculates and returns the area of the wind farm based on the specified polygon.
    """
    if (polygon := params.assemble.polygon) is not None:
        points = convert_to_points(polygon.points)
        polygon = get_windfarm_centered_polygon(points)
        return round(polygon.area / 1e6, 2)  # km^2


def calculate_aep(params, **kwargs):
    """
    Calculates and returns the Annual Energy Production (AEP) for the wind farm.
    """
    if (polygon := params.assemble.polygon) is not None:
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing
        points = convert_to_points(polygon.points)

        # model
        wind_farm = get_wind_farm_model(points, turbine_type)

        # windspeed and direction
        wind_direction = params.visualize.wind_direction
        wind_speed = params.visualize.wind_speed

        # initialize turbine positions
        x, y = get_initial_turbine_positions(points, turbine_type, turbine_spacing)

        # calculate aep
        aep = wind_farm.aep(x, y, wd=wind_direction, ws=wind_speed) / 1e6

        return round(aep, 2)


def calculate_loss(params, **kwargs):
    """
    Calculates and returns the energy loss due to wake effects in the wind farm.
    """
    if (polygon := params.assemble.polygon) is not None:
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing
        points = convert_to_points(polygon.points)

        # model
        wind_farm = get_wind_farm_model(points, turbine_type)

        # windspeed and direction
        wind_direction = params.visualize.wind_direction
        wind_speed = params.visualize.wind_speed

        # initialize turbine positions
        x, y = get_initial_turbine_positions(points, turbine_type, turbine_spacing)

        # calculate aep loss
        aep = wind_farm.aep(x, y, wd=wind_direction, ws=wind_speed) / 1e6
        aep_without_loss = (
            wind_farm.aep(x, y, wd=wind_direction, ws=wind_speed, with_wake_loss=False)
            / 1e6
        )
        loss = (aep_without_loss - aep) / aep_without_loss * 100

        return round(loss, 2)


def number_of_turbines(params, **kwargs):
    """
    Calculates and returns the number of turbines that can be placed within the polygon.
    """
    if (polygon := params.assemble.polygon) is not None:
        points = convert_to_points(polygon.points)
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing

        return get_number_of_turbines(points, turbine_type, turbine_spacing)


def get_wind_farm_model(points, turbine_type) -> WindFarmModel:
    """
    Sets up and returns a wind farm model based on the provided points and turbine type.
    """
    wind_turbine = TURBINE_CLASSES_DICT[turbine_type]()

    # obtain wind distribution data (a.k.a. weibull parameters)
    lat, lon = get_windfarm_centroid_wgs(points)
    wind_data = get_gwc_data(lat, lon)
    heights = list(wind_data.get_index("height"))
    try:
        height_index = heights.index(WIND_MEASUREMENT_HEIGHT)
    except ValueError:
        vkt.UserError("height not avalaible at this site, pick another site")

    # simplify stuff by choosing wind data from a specific height (not very accurate)
    f = wind_data.data_vars.get("frequency")[ROUGHNESS_INDEX].data
    A = wind_data.data_vars.get("A")[ROUGHNESS_INDEX, height_index].data
    k = wind_data.data_vars.get("k")[ROUGHNESS_INDEX, height_index].data

    # default wind directions
    wind_directions = np.linspace(0, 360, 12, endpoint=False)

    # assemble site
    site = XRSite(
        ds=xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", f),
                "Weibull_A": ("wd", A),
                "Weibull_k": ("wd", k),
                "TI": TURBULENCE_INTENSITY,
            },
            coords={"wd": wind_directions},
        )
    )

    # most recent model from literature with recommended turbulence model
    return Blondel_Cathelain_2020(site, wind_turbine, turbulenceModel=CrespoHernandez())


@vkt.memoize
def optimize_turbine_positions(points, turbine_type, turbine_spacing, maxiter):
    """
    Optimizes turbine positions in the wind farm and returns relevant data including
    timestamps, AEP values, and serialized plots of the optimized positions and convergence.
    """
    print(points, turbine_type, turbine_spacing, maxiter)
    # wind farm model
    wind_farm = get_wind_farm_model(points, turbine_type)
    site = wind_farm.site

    # optimized positions plot component
    optimized_positions_fig, optimized_positions_ax = plt.subplots()
    plot_component = XYPlotComp(ax=optimized_positions_ax)

    # construct top farm problem
    topfarm_problem = get_topfarm_problem(
        wind_farm,
        points,
        turbine_type,
        turbine_spacing,
        plot_component,
        maxiter=maxiter,
    )

    # perform optimization routine
    _, _, recorder = topfarm_problem.optimize(disp=False)

    # convergence info
    t, aep = [recorder[v] for v in ["timestamp", "AEP"]]

    # meta info
    n_wt = get_number_of_turbines(points, turbine_type, turbine_spacing)
    n_wd = len(site.default_wd)
    n_ws = len(site.default_ws)

    # save optimized positions plot
    optimized_positions_ax.set_title("")
    optimized_positions_png = vkt.File()
    optimized_positions_fig.savefig(
        optimized_positions_png.source, format="png", dpi=IMAGE_DPI
    )
    plt.close(optimized_positions_fig)

    # convergence plot
    convergence_fig = plt.figure()
    plt.plot(t - t[0], aep / 1e6)
    setup_plot(
        ylabel=r"AEP ($\times 10^6\ \text{GWh}$)",
        xlabel=r"Computation time (s)",
        title=f"{n_wt} wind turbines, {n_wd} wind directions, {n_ws} wind speeds",
    )
    plt.ticklabel_format(useOffset=False)
    convergence_png = vkt.File()
    convergence_fig.savefig(convergence_png.source, format="png", dpi=IMAGE_DPI)
    plt.close(convergence_fig)

    # make sure output is serializeable
    t = list(t)
    aep = list(aep)
    optimized_positions_png_s = serialize(optimized_positions_png)
    convergence_png_s = serialize(convergence_png)

    return (t, aep, convergence_png_s, optimized_positions_png_s)


def get_topfarm_problem(wind_farm, points, turbine_type, turbine_spacing, plot_component, grad_method=autograd, maxiter=4, n_cpu=1) -> TopFarmProblem:
    """
    Creates and returns a TopFarmProblem for optimizing wind turbine positions.
    """
    site = wind_farm.site
    x, y = get_initial_turbine_positions(points, turbine_type, turbine_spacing)
    boundary_points = get_windfarm_centered_points(points)
    boundary_constr = XYBoundaryConstraint(boundary_points, boundary_type="polygon")
    return TopFarmProblem(
        design_vars={"x": x, "y": y},
        cost_comp=PyWakeAEPCostModelComponent(
            wind_farm,
            n_wt=len(x),
            grad_method=grad_method,
            n_cpu=n_cpu,
            wd=site.default_wd,
            ws=site.default_ws,
        ),
        driver=EasyScipyOptimizeDriver(maxiter=maxiter, disp=False),
        constraints=[
            boundary_constr,
            SpacingConstraint(
                min_spacing=get_turbine_spacing(turbine_type, turbine_spacing)
            ),
        ],
        plot_comp=plot_component,
    )


def get_windfarm_boundary(points: list[vkt.Point]):
    """
    Returns the boundary coordinates of the wind farm polygon.
    """
    polygon = get_windfarm_centered_polygon(points)
    return polygon.exterior.xy


def get_buffer_bounds(points: list[vkt.Point]):
    """
    Returns the buffered (extended) bounds of the wind farm polygon.
    """
    polygon = get_windfarm_centered_polygon(points)
    buffer_fraction = 0.05
    buffer_distance = buffer_fraction * polygon.length
    return polygon.buffer(buffer_distance).bounds


def get_number_of_turbines(points, turbine_type, turbine_spacing):
    """
    Returns the number of turbines that can be placed within the wind farm polygon.
    """
    x, _ = get_initial_turbine_positions(points, turbine_type, turbine_spacing)
    return len(x)


def get_initial_turbine_positions(points, turbine_type, turbine_spacing):
    """
    Generates and returns the initial turbine positions within the wind farm polygon.
    """
    polygon = get_windfarm_centered_polygon(points)

    # get bounding box coordinates and lengths
    minx, miny, maxx, maxy = polygon.bounds

    # get turbine spacing
    turbine_spacing = get_turbine_spacing(turbine_type, turbine_spacing)

    # generate uniform grid of turbines in bounding box
    xs = np.arange(minx, maxx, turbine_spacing)
    ys = np.arange(miny, maxy, turbine_spacing)
    x, y = np.meshgrid(xs, ys)
    x, y = x.flatten(), y.flatten()
    grid_points = np.vstack((x, y)).T

    # generate mask for points within polygon
    path = MPLPath(get_windfarm_centered_points(points))
    mask = path.contains_points(grid_points)
    return x[mask], y[mask]


def get_windfarm_centered_polygon(points: list[vkt.Point]):
    """
    Returns the wind farm polygon centered at the centroid of the provided points.
    """
    return ShapelyPolygon(get_windfarm_centered_points(points))


def get_windfarm_centered_points(points: list[vkt.Point]):
    """
    Centers the provided points around the centroid of the wind farm and returns them.
    """
    points_rd = np.array(points)
    centroid_rd = get_windfarm_centroid_rd(points)
    return points_rd - centroid_rd


def get_turbine_spacing(turbine_type, turbine_spacing):
    """
    Calculates and returns the spacing between turbines based on the turbine type and spacing factor.
    """
    turbine = TURBINE_CLASSES_DICT[turbine_type]()
    diameter = turbine.diameter()
    return diameter * turbine_spacing


def get_windfarm_centroid_wgs(points: list[vkt.Point]):
    """
    Returns the centroid of the wind farm in WGS (World Geodetic System) coordinates.
    """
    centroid_rd = get_windfarm_centroid_rd(points)
    return vkt.RDWGSConverter.from_rd_to_wgs(centroid_rd)


def get_windfarm_centroid_rd(points: list[vkt.Point]):
    """
    Returns the centroid of the wind farm in RD (Rijksdriehoek) coordinates.
    """
    return vkt.Polygon([vkt.Point(*point) for point in points]).centroid


def convert_to_points(geo_points: list[vkt.GeoPoint]):
    """
    Converts a list of GeoPoint objects to a list of RD coordinate points.
    """
    return [geo_point.rd for geo_point in geo_points]
