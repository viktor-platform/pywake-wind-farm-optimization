from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt

# PyWake
from py_wake import HorizontalGrid

# VIKTOR
from viktor import File, ViktorController
from viktor.errors import UserError
from viktor.result import ImageResult, OptimizationResult, OptimizationResultElement
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

from lib.constants import IMAGE_DPI, deserialize
from lib.parametrization import Parametrization

# wind farm model
from lib.wind_farm import (
    SITE_MAXIMUM_AREA,
    SITE_MINIMUM_AREA,
    calculate_aep,
    calculate_loss,
    convert_to_points,
    get_buffer_bounds,
    get_initial_turbine_positions,
    get_wind_farm_model,
    get_windfarm_area,
    get_windfarm_boundary,
    get_windfarm_centroid_wgs,
    number_of_turbines,
    optimize_turbine_positions,
)


class Controller(ViktorController):
    label = "wind farm"
    parametrization = Parametrization

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
            points = convert_to_points(params.assemble.polygon.points)
            lat, lon = get_windfarm_centroid_wgs(points)
            features += [MapPoint(lat, lon)]

        return MapResult(features)

    @ImageView("Wind rose", duration_guess=5)
    def wind_rose(self, params, **kwargs):
        # gather data
        points = convert_to_points(params.assemble.polygon.points)
        turbine_type = params.visualize.turbine
        windfarm = get_wind_farm_model(points, turbine_type)

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
        # gather data
        points = convert_to_points(params.assemble.polygon.points)
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing

        # wind farm model
        windfarm = get_wind_farm_model(points, turbine_type)

        # initiliaze turbine positions
        x, y = get_initial_turbine_positions(points, turbine_type, turbine_spacing)

        # windspeed and direction
        wind_direction = params.visualize.wind_direction
        wind_speed = params.visualize.wind_speed

        # simulation
        windfarm_simulated = windfarm(x, y, wd=wind_direction, ws=wind_speed)

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
        xb, yb = get_windfarm_boundary(points)
        plt.plot(xb, yb)

        # zoom to wind farm
        minx, miny, maxx, maxy = get_buffer_bounds(points)
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)

        # save and close
        png = File()
        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()

        # AEP, loss and number of turbines
        data = DataGroup(
            DataItem("AEP", value=calculate_aep(params), suffix="10^6 GWh"),
            DataItem("Loss", value=calculate_loss(params), suffix="%"),
            DataItem("Number of turbines", value=number_of_turbines(params)),
        )
        return ImageAndDataResult(png, data)

    @ImageAndDataView("Optimized positions", duration_guess=5)
    def optimized_positions(self, params, **kwargs):
        # gather params
        points = convert_to_points(params.assemble.polygon.points)
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing

        # optimize positions
        _, aep, _, optimized_positions_png_s = optimize_turbine_positions(
            points,
            turbine_type,
            turbine_spacing,
            maxiter=params.optimize.number_of_iterations,
        )

        # save (increase of) AEP
        increase = (aep[-1] - aep[0]) / aep[0] * 100
        aep_data = {"aep": aep[-1] / 1e6, "increase": increase}

        # collect data group
        aep_data_group = DataGroup(
            DataItem(
                "AEP (optimal)",
                value=aep_data["aep"],
                suffix="10^6 GWh",
                number_of_decimals=2,
            ),
            DataItem(
                "AEP (increase)",
                value=aep_data["increase"],
                suffix="%",
                number_of_decimals=2,
            ),
        )

        optimized_positions_png = deserialize(optimized_positions_png_s)
        return ImageAndDataResult(
            optimized_positions_png,
            aep_data_group,
        )

    def optimization_routine(self, params, **kwargs):
        """
        Optimize wind turbine positions in windfarm.
        """
        # gather data
        points = convert_to_points(params.assemble.polygon.points)
        turbine_type = params.visualize.turbine
        turbine_spacing = params.visualize.turbine_spacing

        # optimize positions
        t, aep, convergence_png_s, _ = optimize_turbine_positions(
            points,
            turbine_type,
            turbine_spacing,
            params.optimize.number_of_iterations,
        )

        # convergence result elements
        results = [
            OptimizationResultElement(
                params, {"time": round(_t - t[0], 2), "aep": round(_aep / 1e6, 3)}
            )
            for _t, _aep in zip(t, aep)
        ]
        output_headers = {"time": "Computation time (s)", "aep": "AEP (10^6 GWh)"}

        # convergence plot
        convergence_png = deserialize(convergence_png_s)

        return OptimizationResult(
            results, output_headers=output_headers, image=ImageResult(convergence_png)
        )
