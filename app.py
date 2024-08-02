from cProfile import label
from locale import normalize
from math import remainder
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MPLPath
from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.lillgrund import SWT23, LillgrundSite
from py_wake.examples.data.ParqueFicticio import ParqueFicticioSite
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.literature.iea37_case_study1 import IEA37CaseStudy1
from py_wake.site import XRSite
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.plotting import setup_plot
from py_wake.wind_farm_models.wind_farm_model import (
    SimulationResult as PyWakeSimulationResult,
)
from py_wake.wind_farm_models.wind_farm_model import (
    WindFarmModel as PyWakeWindFarmModel,
)
from scipy.optimize import least_squares
from shapely import MultiPoint
from shapely.geometry import Polygon as ShapelyPolygon
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import (
    CircleBoundaryConstraint,
    XYBoundaryConstraint,
)
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from viktor import File, ViktorController
from viktor.errors import UserError
from viktor.geometry import Point, Polygon, RDWGSConverter
from viktor.parametrization import (  # FunctionLookup,; Lookup,
    GeoPolygonField,
    LineBreak,
    NumberField,
    OptimizationButton,
    OptionField,
    Step,
    Text,
    ViktorParametrization,
)
from viktor.result import ImageResult, OptimizationResult, OptimizationResultElement
from viktor.views import ImageView, MapPoint, MapPolygon, MapResult, MapView

from gwa_reader import get_gwc_data

#############
# CONSTANTS #
#############
ROOT = Path(__file__).parent
IMAGE_DPI = 800

# windfarm sites
SITES = ["Horns Rev", "Parque Ficticio", "Lillgrund Wind Farm"]
SITE_CLASSES = [Hornsrev1Site, ParqueFicticioSite, LillgrundSite]
SITE_CLASSES_DICT = {site: site_cls for site, site_cls in zip(SITES, SITE_CLASSES)}

# wind turbines
TURBINES = ["V80", "IEA37", "DTU10MW", "SWT 2.3"]
TURBINE_CLASSES = [V80, IEA37_WindTurbines, DTU10MW, SWT23]
TURBINE_CLASSES_DICT = {
    turbine: turbine_cls for turbine, turbine_cls in zip(TURBINES, TURBINE_CLASSES)
}

# model constants
ROUGHNESS_INDEX = 0  # we assume a flat ground (reasonable for off-shore farms)
WIND_MEASUREMENT_HEIGHT = 100.0  # m (for convenience we fix this height. Not accurate as turbine hub heights may vary)
TURBULENCE_INTENSITY = 0.1


def get_divisors(n, min=5):
    bin_nums = np.arange(5, n + 1)
    remainders = np.remainder(n, bin_nums)
    mask = remainders == 0
    return bin_nums[mask].tolist()


WIND_BIN_NUMS = get_divisors(360)


class Parametrization(ViktorParametrization):
    # TODO: Process advice from Stijn & Matthijs about making things sound less technical: "Make things sound sexy..."
    assemble = Step("Assemble wind farm", views=["site_locations", "wind_rose"])
    assemble.welcome_text = Text("# Welcome to wind farm modelling with PyWake! 💨")
    assemble.polygon = GeoPolygonField("Mark site")
    assemble.site = OptionField("Choose site", options=SITES, default=SITES[0])

    assemble.turbine = OptionField(
        "turbine type", options=TURBINES, default=TURBINES[0]
    )
    assemble.wind_rose_header = Text("# Wind Rose")
    assemble.number_wind_directions = OptionField(
        "number of wind direction bins",
        options=WIND_BIN_NUMS,
        default=WIND_BIN_NUMS[-1],
    )
    assemble.number_wind_speeds = NumberField(
        "number of wind speed bins", variant="slider", min=2, max=4, default=4
    )

    inspect = Step("Inspect wakes", views="wake_plot")
    inspect.number_of_turbines = NumberField(
        "number of turbines",
        default=9,
        min=1,
        max=100,
    )
    inspect.turbine_spacing = NumberField(
        "minimum spacing between turbines",
        default=7,
        min=2,
        max=15,
        suffix="turbine diameters",
        variant="slider",
        step=1,
    )
    inspect.wind_direction = NumberField(
        "Wind direction",
        min=0,
        max=359,
        suffix="°",
        variant="slider",
        step=1,
        default=270,
        # step=FunctionLookup(
        #     GET_WIND_DIRECTION_STEP_SIZE, Lookup("assemble.wind_direction_resolution")
        # )
    )
    inspect.wind_speed = NumberField(
        "Wind speed",
        min=4,
        max=30,
        suffix="m/s",
        variant="slider",
        step=0.1,
        default=10,
    )

    optimize = Step("Optimize positions", views="optimal_aep_per_turbine")
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
    def site_locations(self, params, **kwargs):
        features = []

        if (polygon := params.assemble.polygon) is not None:
            features += [
                MapPolygon.from_geo_polygon(polygon),
            ]
            centroid = self._get_windfarm_centroid(params)
            lat, lon = RDWGSConverter.from_rd_to_wgs(centroid)
            features += [MapPoint(lat, lon)]

        return MapResult(features)

    @ImageView("Wind rose", duration_guess=1)
    def wind_rose(self, params, **kwargs):
        windfarm = self.get_wind_farm_model(params, **kwargs)

        # wind rose plot
        fig = plt.figure()
        png = File()
        _ = windfarm.site.plot_wd_distribution(
            n_wd=params.assemble.number_wind_directions,
            ws_bins=params.assemble.number_wind_speeds + 1,
        )  # TODO: add height option?

        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        return ImageResult(png)

    @ImageView("Wake plot", duration_guess=1)
    def wake_plot(self, params, **kwargs):
        windfarm = self.get_wind_farm_model(params, **kwargs)

        # initiliaze turbine positions
        x, y = self._get_initial_turbine_positions(params)

        # windspeed and direction
        wind_direction = params.inspect.wind_direction
        wind_speed = params.inspect.wind_speed

        # simulation
        windfarm_simulated = windfarm(x, y, wd=wind_direction, ws=wind_speed)

        # define flow map
        grid = HorizontalGrid(x=None, y=None, resolution=300, extend=1.5)
        flow_map = windfarm_simulated.flow_map(
            grid=grid, wd=wind_direction, ws=wind_speed
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
        minx, miny, maxx, maxy = self._get_buffer_bounds(params)
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)

        # save and close
        png = File()
        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        return ImageResult(png)

    @ImageView("Wake plot", duration_guess=1)
    def optimal_aep_per_turbine(self, params, **kwargs):
        png = File().from_path(ROOT / "lib" / "optimization_functionality_sample.png")
        return ImageResult(png)

    #########
    # MODEL #
    #########
    def get_wind_farm_model(self, params, **kwargs) -> PyWakeWindFarmModel:
        """
        Setup wind farm model.
        """
        wind_turbine = TURBINE_CLASSES_DICT[params.assemble.turbine]()

        # weibull parameters
        centroid = self._get_windfarm_centroid(params, **kwargs)
        lat, lon = RDWGSConverter.from_rd_to_wgs(centroid)
        wind_data = get_gwc_data(lat, lon)
        heights = list(wind_data.get_index("height"))
        try:
            height_index = heights.index(WIND_MEASUREMENT_HEIGHT)
        except ValueError:
            UserError("height not avalaible at this site")

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

        # construct top farm problem
        topfarm_problem = self.get_topfarm_problem(params, wind_farm)

        # perform optimization routine (~ 60s)
        cost, state, recorder = topfarm_problem.optimize(disp=True)

        # convergence info
        t, aep = [recorder[v] for v in ["timestamp", "AEP"]]

        # meta info
        x, _ = self._get_initial_turbine_positions(params)
        n_wt = len(x)
        n_wd = len(site.default_wd)
        n_ws = len(site.default_ws)

        # convergence results
        results = []
        output_headers = {}
        print("AEP ", aep)

        # convergence plot
        fig = plt.figure()
        plt.plot(t - t[0], aep, label="...")
        setup_plot(
            ylabel="AEP [GWh]",
            xlabel="Time [s]",
            title=f"{n_wt} wind turbines, {n_wd} wind directions, {n_ws} wind speeds",
        )
        plt.ticklabel_format(useOffset=False)
        png = File()
        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        image = ImageResult(png)

        return OptimizationResult(results, output_headers=output_headers, image=image)

    def get_topfarm_problem(
        self, params, wind_farm, grad_method=autograd, maxiter=4, n_cpu=1, **kwargs
    ) -> TopFarmProblem:
        """
        function to create a topfarm problem, following the elements of OpenMDAO architecture
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
        )

    ##############
    # SUPPORTING #
    ##############

    def _get_windfarm_boundary(self, params, **kwargs):
        polygon = self._get_windfarm_polygon(params)
        return polygon.exterior.xy

    def _get_windfarm_centroid(self, params, **kwargs):
        points = [Point(*point) for point in self._get_windfarm_points(params)]
        return Polygon(points).centroid

    def _get_windfarm_points(self, params, **kwargs):
        if (polygon := params.assemble.polygon) is not None:
            return [point.rd for point in polygon.points]
        else:
            raise UserError("First specify wind farm polygon")

    def _get_initial_turbine_positions(self, params, **kwargs):
        polygon = self._get_windfarm_polygon(params)

        # get bounding box coordinates and lengths
        minx, miny, maxx, maxy = polygon.bounds

        # get turbine diameter
        turbine_spacing = self._get_turbine_spacing(params)

        # generate uniform grid of turbines in bounding box
        xs = np.arange(minx, maxx, turbine_spacing)
        ys = np.arange(miny, maxy, turbine_spacing)

        x, y = np.meshgrid(xs, ys)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        # select only points within polygon
        path = self._get_windfarm_path(params)
        mask = path.contains_points(points)

        print("number turbines = ", len(x[mask]))

        return x[mask], y[mask]

    def _get_buffer_bounds(self, params, **kwargs):
        polygon = self._get_windfarm_polygon(params)
        buffer_fraction = 0.05  # fraction of total distance
        buffer_distance = buffer_fraction * polygon.length
        return polygon.buffer(buffer_distance).bounds

    def _get_windfarm_polygon(self, params, **kwargs):
        points = np.array(self._get_windfarm_points(params))
        centroid = self._get_windfarm_centroid(params)

        # translate polygon to origin for convenience
        return ShapelyPolygon(points - centroid)

    def _get_windfarm_path(self, params, **kwargs):
        points = np.array(self._get_windfarm_points(params))
        centroid = self._get_windfarm_centroid(params)
        return MPLPath(points - centroid)

    def _get_turbine_spacing(self, params):
        turbine = TURBINE_CLASSES_DICT[params.assemble.turbine]()
        diameter = turbine.diameter()
        return diameter * params.inspect.turbine_spacing
