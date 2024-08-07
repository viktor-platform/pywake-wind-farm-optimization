from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=RuntimeWarning)

from json import dumps, loads
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MPLPath

# PyWake
from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import V80
from py_wake.examples.data.iea37 import IEA37_WindTurbines
from py_wake.examples.data.lillgrund import SWT23
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.site import XRSite
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.gradients import autograd
from py_wake.utils.plotting import setup_plot
from py_wake.wind_farm_models.wind_farm_model import (
    WindFarmModel as PyWakeWindFarmModel,
)

# Shapely
from shapely.geometry import Polygon as ShapelyPolygon

# Topfarm
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp

# VIKTOR
from viktor import File, ViktorController
from viktor.core import Storage
from viktor.errors import UserError
from viktor.geometry import Point, Polygon, RDWGSConverter
from viktor.parametrization import (
    GeoPolygonField,
    Image,
    IsNotNone,
    Lookup,
    NumberField,
    OptimizationButton,
    OptionField,
    OutputField,
    Step,
    Text,
    ViktorParametrization,
)
from viktor.result import ImageResult, OptimizationResult, OptimizationResultElement
from viktor.utils import memoize
from viktor.views import (
    DataGroup,
    DataItem,
    ImageAndDataResult,
    ImageAndDataView,
    ImageView,
    MapPoint,
    MapPolygon,
    MapResult,
    MapView,
)

# Global Wind Atlas API
from gwa_reader import get_gwc_data

#############
# CONSTANTS #
#############
ROOT = Path(__file__).parent
IMAGE_DPI = 800


def get_divisors(n, minimum=5):
    bin_nums = np.arange(minimum, n + 1)
    remainders = np.remainder(n, bin_nums)
    mask = remainders == 0
    return bin_nums[mask].tolist()


# windfarm sites
SITE_MINIMUM_AREA = 1  # km^2
SITE_MAXIMUM_AREA = 20  # km^2


def get_windfarm_area(params, **kwargs):
    if params.assemble.polygon:
        return round(Controller._get_windfarm_polygon(params).area / 1e6, 2)  # km^2


# wind turbines
TURBINES = ["V80 (2.0 MW)", "SWT (2.3 MW)" "IEA37 (3.35 MW)", "DTU (10 MW)"]
TURBINE_CLASSES = [V80, SWT23, IEA37_WindTurbines, DTU10MW]
TURBINE_CLASSES_DICT = {
    turbine: turbine_cls for turbine, turbine_cls in zip(TURBINES, TURBINE_CLASSES)
}

# physical model
ROUGHNESS_INDEX = 0  # we assume a flat ground (reasonable for off-shore farms)
WIND_MEASUREMENT_HEIGHT = 100.0  # m (for convenience we fix this height. Not accurate as turbine hub heights may vary)
TURBULENCE_INTENSITY = 0.1
WIND_BIN_NUMS = get_divisors(360)


# @memoize
# def simulate_wind_farm(, x, y, wind_speed, wind_direction):
#     return windfarm(x, y, wd=wind_speed, ws=wind_direction)


def get_simulated_wind_farm(params, **kwargs):
    # wind farm model
    windfarm = Controller.get_wind_farm_model(params)

    # initiliaze turbine positions
    x, y = Controller._get_initial_turbine_positions(params)

    # windspeed and direction
    wind_direction = params.visualize.wind_direction
    wind_speed = params.visualize.wind_speed

    # simulation
    windfarm_simulated = windfarm(x, y, wd=wind_direction, ws=wind_speed)

    return windfarm_simulated


def calculate_aep(params, **kwargs):
    if params.assemble.polygon:
        simulated_wind_farm = get_simulated_wind_farm(params)
        aep = simulated_wind_farm.aep().sum().data / 1e6

        return round(aep, 2)


def calculate_loss(params, **kwargs):
    if params.assemble.polygon:
        simulated_wind_farm = get_simulated_wind_farm(params)
        aep = simulated_wind_farm.aep().sum().data
        aep_without_loss = simulated_wind_farm.aep(with_wake_loss=False).sum().data
        loss = (aep_without_loss - aep) / aep_without_loss * 100

        if loss < 0:  # corrcet for slightly negative values
            loss = 0
        return round(loss, 2)


def number_of_turbines(params, **kwargs):
    if params.assemble.polygon:
        x, _ = Controller._get_initial_turbine_positions(params)
        return len(x)


class Parametrization(ViktorParametrization):
    assemble = Step("Select location", views=["site_location"], width=30)
    assemble.welcome_text = Text(
        """
# Welcome to wind farm modelling with PyWake! 💨
With this app you can design a wind farm in a few steps.

Start by selecting an area for your wind farm by drawing a polygon on the map
        """
    )
    assemble.polygon = GeoPolygonField("")
    assemble.windfarm_area = OutputField(
        "Wind farm area", suffix=r"$\textrm{km}^2$", value=get_windfarm_area, flex=50
    )

    _polygon_selected = IsNotNone(Lookup("assemble.polygon"))
    conditions = Step("Wind conditions", views="wind_rose", enabled=_polygon_selected)
    conditions.text = Text(
        """
# Wind conditions at your site
Wind conditions are gathered at your site location from the [Global Wind Atlas](https://globalwindatlas.info/en/)

Below you can edit the wind rose plot to your liking.
        """
    )
    conditions.number_wind_directions = OptionField(
        "number of wind direction bins",
        options=WIND_BIN_NUMS,
        default=WIND_BIN_NUMS[-1],
    )
    conditions.number_wind_speeds = NumberField(
        "number of wind speed bins", variant="slider", min=2, max=4, default=4
    )

    visualize = Step("Visualize wakes", views="wake_plot", enabled=_polygon_selected)
    visualize.intro_text = Text(
        """
# Visualize your windfarm
Update the view on the right to view your wind farm's layout. 
The turbines are automatically placed in a grid based on some minimum spacing. Keep reading below and
find out how you can improve the current layout!

## Wake effects
Wake effects limit how much energy a wind farm produces. 
Below you can try to account for wake effects in your wind farm. 
    """
    )
    visualize.wake_effects_image = Image(path="wake_effects.jpg")
    visualize.revenue_text = Text(
        """

The goal is to maximize your wind farm's Annual Energy Production (AEP), 
while using a relatively low number of turbines. These values play an essential 
role in determining when you can expect a Return on Investment (ROI) of your wind farm. 
The output fields below as well the data menu in the wake plot (press the "<" on the right) 
show (AEP) and the number of turbines.
"""
    )
    visualize.aep = OutputField("AEP", value=calculate_aep, suffix=r"$\textrm{GW}$")
    visualize.loss = OutputField("Loss", value=calculate_loss, suffix="%")
    visualize.number_of_turbines = OutputField(
        "Number of turbines", value=number_of_turbines
    )
    visualize.wind_velocity_text = Text(
        """
## Wind
Wind direction and -speed will change troughout the course of a year. However, you 
can get a good idea of what the typical wind conditions will be at your site by studying the wind rose
you generated in the 'Wind conditions' step. Try setting the wind direction and -speed to the most 
common values you gather from the wind rose. 
    """
    )
    visualize.wind_direction = NumberField(
        "Wind direction",
        min=0,
        max=359,
        suffix="°",
        variant="slider",
        step=1,
        default=270,
    )
    visualize.wind_speed = NumberField(
        "Wind speed",
        min=4,
        max=30,
        suffix="m/s",
        variant="slider",
        step=0.1,
        default=10,
    )
    visualize.wind_park_layout_text = Text(
        """
## Turbines 
Try to keep the follwing in mind, while designing your wind farm:
- Opting for a bigger turbine increases the energy production per individual turbine, 
but allows for a lower total number of turbines and increases wake effects;
- Additionally, turbines are typically spaced several multiples of their diameter away from each other. 
Increasing this spacing reduces wake effects as well as the total number of turbines and AEP. 
"""
    )
    visualize.turbine = OptionField("Type", options=TURBINES, default=TURBINES[0])
    visualize.turbine_spacing = NumberField(
        "Spacing",
        default=7,
        min=2,
        max=15,
        suffix="turbine diameters",
        variant="slider",
        step=1,
    )

    optimize = Step(
        "Optimize turbine locations",
        views="optimized_positions",
        enabled=_polygon_selected,
    )
    optimize.text = Text(
        """
# Optimize your wind farm

Many factors come into play when optmizing your wind farm's layout. The previous step
illustrates how this can complicate finding the most efficient and profitable wind farm.
Luckily, we can use the [Topfarm module](https://topfarm.pages.windenergy.dtu.dk/TopFarm2/) 
to automatically find improved positions for the wind turbines. 

By pressing the optimization button below you can further improve on your wind farm's layout.
After the routine has finished, you can update the "Optimized positions" view on the right to see 
the recommended changes!
        """
    )
    optimize.positions = OptimizationButton(
        "Optimize turbine positions", "optimize_turbine_positions", longpoll=True
    )


class Controller(ViktorController):
    label = "wind farm"
    parametrization = Parametrization

    #########
    # VIEWS #
    #########
    @MapView("Site map", duration_guess=1)
    def site_location(self, params, **kwargs):
        features = []

        if (polygon := params.assemble.polygon) is not None:
            area = get_windfarm_area(params)
            if area < SITE_MINIMUM_AREA:
                raise UserError(
                    f"Choose a larger site. Current: {area:.1f} "
                    + r"(km^2). "
                    + f"Required: {SITE_MINIMUM_AREA} < A < {SITE_MAXIMUM_AREA} (km^2)"
                )
            if area > SITE_MAXIMUM_AREA:
                raise UserError(
                    f"Choose a smaller site. Current: {area:.1f}"
                    + r"(km^2). "
                    + f"Required: {SITE_MINIMUM_AREA} < A < {SITE_MAXIMUM_AREA} (km^2)"
                )
            features += [
                MapPolygon.from_geo_polygon(polygon),
            ]
            centroid = self._get_windfarm_centroid(params)
            lat, lon = RDWGSConverter.from_rd_to_wgs(centroid)
            features += [MapPoint(lat, lon)]

        return MapResult(features)

    @ImageView("Wind rose", duration_guess=5)
    def wind_rose(self, params, **kwargs):
        windfarm = self.get_wind_farm_model(params, **kwargs)

        # wind rose plot
        fig = plt.figure()
        png = File()
        _ = windfarm.site.plot_wd_distribution(
            n_wd=params.conditions.number_wind_directions,
            ws_bins=params.conditions.number_wind_speeds + 1,
        )  # TODO: add height option?

        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        return ImageResult(png)

    @ImageAndDataView("Wake plot", duration_guess=5)
    def wake_plot(self, params, **kwargs):
        # simulate wind farm
        windfarm_simulated = get_simulated_wind_farm(params)

        # define flow map
        grid = HorizontalGrid(x=None, y=None, resolution=300, extend=1.5)
        flow_map = windfarm_simulated.flow_map(
            grid=grid,
            wd=params.visualize.wind_direction,
            ws=params.visualize.wind_speed,
        )

        # wake plot
        fig = plt.figure()
        flow_map.plot_wake_map()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        # wind farm boundary
        xb, yb = self._get_windfarm_boundary(params)
        plt.plot(xb, yb)

        # zoom to wind farm
        minx, miny, maxx, maxy = Controller._get_buffer_bounds(params)
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)

        # save and close
        png = File()
        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()

        # AEP, loss and number of turbines
        data = DataGroup(
            DataItem("AEP", value=calculate_aep(params), suffix="GW"),
            DataItem("Loss", value=calculate_loss(params), suffix="%"),
            DataItem("Number of turbines", value=number_of_turbines(params)),
        )
        return ImageAndDataResult(png, data)

    @ImageAndDataView("Optimized positions", duration_guess=5)
    def optimized_positions(self, params, **kwargs):
        with open(ROOT / "lib" / "optimized_positions_aep", "rb") as aep_data_f:
            aep_data = loads(aep_data_f.read().decode("utf-8"))
            aep_data_group = DataGroup(
                DataItem(
                    "AEP (optimal)",
                    value=aep_data["aep"],
                    suffix="GW",
                    number_of_decimals=2,
                ),
                DataItem(
                    "AEP (increase)",
                    value=aep_data["increase"],
                    suffix="%",
                    number_of_decimals=2,
                ),
            )
        with open(ROOT / "lib" / "optimized_positions_plot.png", "rb") as png_f:
            png = File.from_data(png_f.read())
            return ImageAndDataResult(png, aep_data_group)

    @ImageView("aep per turbine", duration_guess=1)
    def optimal_aep_per_turbine(self, params, **kwargs):
        png = File().from_path(ROOT / "lib" / "optimization_functionality_sample.png")
        return ImageResult(png)

    #########
    # MODEL #
    #########
    @staticmethod
    def get_wind_farm_model(params, **kwargs) -> PyWakeWindFarmModel:
        """
        Setup wind farm model.
        """
        wind_turbine = TURBINE_CLASSES_DICT[params.visualize.turbine]()

        # obtain wind distribution data (a.k.a. weibull parameters)
        centroid = Controller._get_windfarm_centroid(params, **kwargs)
        lat, lon = RDWGSConverter.from_rd_to_wgs(centroid)
        wind_data = get_gwc_data(lat, lon)
        heights = list(wind_data.get_index("height"))
        try:
            height_index = heights.index(WIND_MEASUREMENT_HEIGHT)
        except ValueError:
            UserError("height not avalaible at this site")

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
        return Blondel_Cathelain_2020(
            site, wind_turbine, turbulenceModel=CrespoHernandez()
        )

    ################
    # OPTIMIZATION #
    ################
    def optimize_turbine_positions(self, params, **kwargs):
        """
        Optimize wind turbine positions in windfarm.
        """
        # collect model
        wind_farm = self.get_wind_farm_model(params)
        site = wind_farm.site

        # optimized positions plot component
        optimized_positions_fig, optimized_positions_ax = plt.subplots()
        plot_component = XYPlotComp(ax=optimized_positions_ax)

        # construct top farm problem
        topfarm_problem = self.get_topfarm_problem(params, wind_farm, plot_component)

        # perform optimization routine (~ 60s)
        _, _, recorder = topfarm_problem.optimize(disp=True)

        # convergence info
        t, aep = [recorder[v] for v in ["timestamp", "AEP"]]

        # meta info
        n_wt = number_of_turbines(params)
        n_wd = len(site.default_wd)
        n_ws = len(site.default_ws)

        # convergence results
        results = [
            OptimizationResultElement(
                params, {"time": round(_t - t[0], 2), "aep": round(_aep / 1e6, 3)}
            )
            for _t, _aep in zip(t, aep)
        ]
        output_headers = {"time": "Time (s)", "aep": "AEP (GWh)"}

        # save optimized positions plot
        optimized_positions_ax.set_title("")
        with open(ROOT / "lib" / "optimized_positions_plot.png", "wb") as png:
            optimized_positions_fig.savefig(png, format="png", dpi=IMAGE_DPI)
        plt.close(optimized_positions_fig)

        # save (increase of) AEP
        with open(ROOT / "lib" / "optimized_positions_aep", "wb") as aep_data_f:
            increase = (aep[-1] - aep[0]) / aep[0] * 100
            data = {"aep": aep[-1] / 1e6, "increase": increase}
            aep_data_f.write(dumps(data).encode("utf-8"))

        # convergence plot
        convergence_fig = plt.figure()
        plt.plot(t - t[0], aep / 1e6)
        setup_plot(
            ylabel="AEP (GWh)",
            xlabel="Time (s)",
            title=f"{n_wt} wind turbines, {n_wd} wind directions, {n_ws} wind speeds",
        )
        plt.ticklabel_format(useOffset=False)
        png = File()
        convergence_fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close(convergence_fig)
        image = ImageResult(png)

        return OptimizationResult(results, output_headers=output_headers, image=image)

    def get_topfarm_problem(
        self,
        params,
        wind_farm,
        plot_component,
        grad_method=autograd,
        maxiter=4,
        n_cpu=1,
        **kwargs,
    ) -> TopFarmProblem:
        """
        function to create a topfarm problem, following the elements of OpenMDAO architecture.
        """
        x, y = self._get_initial_turbine_positions(params)
        centroid = self._get_windfarm_centroid(params)
        boundary_points = np.array(self._get_windfarm_points(params)) - centroid
        boundary_constr = XYBoundaryConstraint(boundary_points, boundary_type="polygon")
        return TopFarmProblem(
            design_vars={"x": x, "y": y},
            cost_comp=PyWakeAEPCostModelComponent(
                wind_farm,
                n_wt=len(x),
                grad_method=grad_method,
                n_cpu=n_cpu,
                wd=wind_farm.site.default_wd,
                ws=wind_farm.site.default_ws,
            ),
            driver=EasyScipyOptimizeDriver(maxiter=maxiter, disp=False),
            constraints=[
                boundary_constr,
                SpacingConstraint(min_spacing=self._get_turbine_spacing(params)),
            ],
            plot_comp=plot_component,
        )

    ##############
    # SUPPORTING #
    ##############
    @staticmethod
    def _get_windfarm_boundary(params, **kwargs):
        polygon = Controller._get_windfarm_polygon(params)
        return polygon.exterior.xy

    @staticmethod
    def _get_buffer_bounds(params, **kwargs):
        polygon = Controller._get_windfarm_polygon(params)
        buffer_fraction = 0.05
        buffer_distance = buffer_fraction * polygon.length
        return polygon.buffer(buffer_distance).bounds

    @staticmethod
    def _get_initial_turbine_positions(params, **kwargs):
        polygon = Controller._get_windfarm_polygon(params)

        # get bounding box coordinates and lengths
        minx, miny, maxx, maxy = polygon.bounds

        # get turbine spacing
        turbine_spacing = Controller._get_turbine_spacing(params)

        # generate uniform grid of turbines in bounding box
        xs = np.arange(minx, maxx, turbine_spacing)
        ys = np.arange(miny, maxy, turbine_spacing)
        x, y = np.meshgrid(xs, ys)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        # generate mask for points within polygon
        path = Controller._get_windfarm_path(params)
        mask = path.contains_points(points)
        return x[mask], y[mask]

    @staticmethod
    def _get_windfarm_polygon(params, **kwargs):
        points = np.array(Controller._get_windfarm_points(params))
        centroid = Controller._get_windfarm_centroid(params)
        return ShapelyPolygon(points - centroid)

    @staticmethod
    def _get_windfarm_path(params, **kwargs):
        points = np.array(Controller._get_windfarm_points(params))
        centroid = Controller._get_windfarm_centroid(params)
        return MPLPath(points - centroid)

    @staticmethod
    def _get_windfarm_centroid(params, **kwargs):
        points = [Point(*point) for point in Controller._get_windfarm_points(params)]
        return Polygon(points).centroid

    @staticmethod
    def _get_windfarm_points(params, **kwargs):
        polygon = params.assemble.polygon
        return [point.rd for point in polygon.points]

    @staticmethod
    def _get_turbine_spacing(params):
        turbine = TURBINE_CLASSES_DICT[params.visualize.turbine]()
        diameter = turbine.diameter()
        return diameter * params.visualize.turbine_spacing
