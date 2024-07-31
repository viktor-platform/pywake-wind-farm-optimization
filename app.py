from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.lillgrund import SWT23, LillgrundSite
from py_wake.examples.data.ParqueFicticio import ParqueFicticioSite
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.literature.iea37_case_study1 import IEA37CaseStudy1
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.plotting import setup_plot
from py_wake.wind_farm_models.wind_farm_model import (
    SimulationResult as PyWakeSimulationResult,
)
from py_wake.wind_farm_models.wind_farm_model import (
    WindFarmModel as PyWakeWindFarmModel,
)
from scipy.optimize import least_squares
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import (
    CircleBoundaryConstraint,
    XYBoundaryConstraint,
)
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from viktor import File, ViktorController
from viktor.parametrization import (  # FunctionLookup,; Lookup,
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

#############
# CONSTANTS #
#############
ROOT = Path(__file__).parent
IMAGE_DPI = 800

# windfarm sites
SITES = ["Horns Rev", "IEA37", "Parque Ficticio", "Lillgrund Wind Farm"]
SITE_CLASSES = [Hornsrev1Site, IEA37Site, ParqueFicticioSite, LillgrundSite]
SITE_CLASSES_DICT = {site: site_cls for site, site_cls in zip(SITES, SITE_CLASSES)}

# wind turbines
TURBINES = ["V80", "IEA37", "DTU10MW", "SWT 2.3"]
TURBINE_CLASSES = [V80, IEA37_WindTurbines, DTU10MW, SWT23]
TURBINE_CLASSES_DICT = {
    turbine: turbine_cls for turbine, turbine_cls in zip(TURBINES, TURBINE_CLASSES)
}

# def GET_WIND_DIRECTION_STEP_SIZE(wind_direction_resolution):
#     return 360 / wind_direction_resolution


class Parametrization(ViktorParametrization):
    assemble = Step("Assemble wind farm", views=["site_locations", "wind_rose"])
    assemble.welcome_text = Text("# Welcome to wind farm modelling with PyWake! 💨")
    assemble.site = OptionField("Choose site", options=SITES, default="IEA37")

    assemble.turbine = OptionField(
        "Choose turbine type", options=TURBINES, default="V80"
    )
    assemble.number_of_turbines = NumberField(
        "Specify number of turbines", default=9, min=1, max=100, visible=False
    )
    assemble.wind_rose_header = Text("# Wind Rose")
    assemble.wind_direction_resolution = NumberField(
        "Specify number of wind direction bins", default=360, min=5
    )

    inspect = Step("Inspect wakes", views="wake_plot")
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
        min=0,
        max=30,
        suffix="m/s",
        variant="slider",
        step=0.1,
        default=10
    )

    optimize = Step("Optimize positions", views="optimal_aep_per_turbine")
    optimize.positions = OptimizationButton(
        "Optimize turbine positions", "optimze_turbine_positions", longpoll=True
    )


class Controller(ViktorController):
    label = "wind farm"
    parametrization = Parametrization

    #########
    # VIEWS #
    #########
    @MapView("Site map", duration_guess=1)
    def site_locations(self, params, **kwargs):
        # TODO: implement site locations. map Below is just placeholder
        # Create some points using coordinates
        markers = [
            MapPoint(25.7617, -80.1918, description="Miami"),
            MapPoint(18.4655, -66.1057, description="Puerto Rico"),
            MapPoint(32.3078, -64.7505, description="Bermudas"),
        ]
        # Create a polygon
        polygon = MapPolygon(markers)

        # Visualize map
        features = markers + [polygon]
        return MapResult(features)

    @ImageView("Wind rose", duration_guess=1)
    def wind_rose(self, params, **kwargs):
        windfarm = self.get_windfarm_model(params, **kwargs)

        # wind rose plot
        fig = plt.figure()
        png = File()
        windspeed_bins = np.arange(0, 41, 5)
        print(windspeed_bins)
        _ = windfarm.site.plot_wd_distribution(n_wd=360)  # TODO: add height option?

        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        return ImageResult(png)

    @ImageView("Wake plot", duration_guess=1)
    def wake_plot(self, params, **kwargs):
        windfarm = self.get_windfarm_model(params, **kwargs)
        x, y = windfarm.site.initial_position.T

        # windspeed and direction
        wind_direction = params.inspect.wind_direction
        wind_speed = params.inspect.wind_speed

        # simulation
        windfarm_simulated = windfarm(x, y, wd=wind_direction, ws=wind_speed)

        # define flow map
        grid = HorizontalGrid(x=None, y=None, resolution=100, extend=1)
        flow_map = windfarm_simulated.flow_map(
            grid=grid, wd=wind_direction, ws=wind_speed
        )

        # wake plot
        fig = plt.figure()
        flow_map.plot_wake_map()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        png = File()
        fig.savefig(
            png.source,
            format="png",
            dpi=IMAGE_DPI
        )
        plt.close()
        return ImageResult(png)

    @ImageView("Wake plot", duration_guess=1)
    def optimal_aep_per_turbine(self, params, **kwargs):
        png = File().from_path(ROOT / "lib" / "optimization_functionality_sample.png")
        return ImageResult(png)

    ################
    # OPTIMIZATION #
    ################
    @staticmethod
    def get_topfarm_problem_xy(
        params, grad_method=autograd, maxiter=100, n_cpu=1, **kwargs
    ) -> TopFarmProblem:
        """
        function to create a topfarm problem, following the elements of OpenMDAO architecture
        """
        windfarm_model = Controller.get_windfarm_model(params)
        x, y = windfarm_model.site.initial_position.T
        boundary_constr = [
            XYBoundaryConstraint(np.array([x, y]).T),
            CircleBoundaryConstraint(
                center=[0, 0], radius=np.round(np.hypot(x, y).max())
            ),
        ][int(isinstance(windfarm_model.site, IEA37Site))]

        return TopFarmProblem(
            design_vars={"x": x, "y": y},
            cost_comp=PyWakeAEPCostModelComponent(
                windFarmModel=windfarm_model,
                n_wt=len(x),
                grad_method=grad_method,
                n_cpu=n_cpu,
                wd=windfarm_model.site.default_wd,
                ws=windfarm_model.site.default_ws,
            ),
            driver=EasyScipyOptimizeDriver(maxiter=maxiter),
            constraints=[
                boundary_constr,
                SpacingConstraint(
                    min_spacing=2 * windfarm_model.windTurbines.diameter()
                ),
            ],
        )

    def optimize_turbine_positions(self, params, **kwargs):
        """
        Optimize wind turbine positions in windfarm.
        """
        results = []
        output_headers = {}
        return OptimizationResult(results, output_headers=output_headers)

    ##############
    # SUPPORTING #
    ##############
    def get_windfarm_model(self, params, **kwargs) -> PyWakeWindFarmModel:
        """
        Setup default windfarm to some initial state.
        """
        # TODO: use params to select the model
        site = IEA37Site(
            params.assemble.number_of_turbines
        )  # TODO: figure out a way to make number of wind turbines variable
        site.default_wd = np.linspace(
            0, 360, params.assemble.wind_direction_resolution, endpoint=False
        )
        wind_turbine = TURBINE_CLASSES_DICT[params.assemble.turbine]
        return Blondel_Cathelain_2020(site, wind_turbine(), turbulenceModel=CrespoHernandez())
        # return Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555)

    @staticmethod
    def calculate_windfarm_aep(
        initial_positions, simulationResult: PyWakeSimulationResult = None
    ) -> float:
        """Calculate wake loss of windfam configuration from AEP."""
        aep = simulationResult.aep().sum().data
        # aepNoWakeLoss = simulationResult.aep(with_wake_loss=False).sum().data
        # return (aep - aepNoWakeLoss) / aepNoWakeLoss
        return aep
