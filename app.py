from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import py_wake
import viktor as vkt

from parametrization import Parametrization
from wind_farm import SITE_MAXIMUM_AREA, SITE_MINIMUM_AREA, IMAGE_DPI, calculate_aep, calculate_loss, convert_to_points, get_buffer_bounds, get_initial_turbine_positions, \
    get_wind_farm_model, get_windfarm_area, get_windfarm_boundary, get_windfarm_centroid_wgs, number_of_turbines, optimize_turbine_positions, deserialize


class Controller(vkt.ViktorController):
    label = "Wind farm"
    parametrization = Parametrization

    @vkt.MapView("Site map", duration_guess=1)
    def site_location(self, params, **kwargs):
        """
        Displays the site location on a map. Validates the area of the site against the minimum
        and maximum allowed values. If valid, displays the polygon representing the site boundary
        and a point indicating the centroid of the wind farm.
        """
        features = []

        if (polygon := params.assemble.polygon) is not None:
            area = get_windfarm_area(params)
            if area < SITE_MINIMUM_AREA:
                raise vkt.UserError(
                    f"Choose a larger site. Current: {area:.1f} "
                    + r"(km^2). "
                    + f"Required: {SITE_MINIMUM_AREA} < A < {SITE_MAXIMUM_AREA} (km^2)"
                )
            if area > SITE_MAXIMUM_AREA:
                raise vkt.UserError(
                    f"Choose a smaller site. Current: {area:.1f}"
                    + r"(km^2). "
                    + f"Required: {SITE_MINIMUM_AREA} < A < {SITE_MAXIMUM_AREA} (km^2)"
                )
            features += [vkt.MapPolygon.from_geo_polygon(polygon)]
            points = convert_to_points(params.assemble.polygon.points)
            lat, lon = get_windfarm_centroid_wgs(points)
            features += [vkt.MapPoint(lat, lon)]

        return vkt.MapResult(features)

    @vkt.ImageView("Wind rose", duration_guess=5)
    def wind_rose(self, params, **kwargs):
        """
        Generates and displays a wind rose plot, showing the distribution of wind direction
        and speed at the wind farm site based on the selected number of wind directions
        and speed bins.
        """
        # gather data
        points = convert_to_points(params.assemble.polygon.points)
        windfarm = get_wind_farm_model(points, params.visualize.turbine)

        # wind rose plot
        fig = plt.figure()
        png = vkt.File()
        _ = windfarm.site.plot_wd_distribution(n_wd=params.conditions.number_wind_directions, ws_bins=params.conditions.number_wind_speeds + 1)

        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()
        return vkt.ImageResult(png)

    @vkt.ImageAndDataView("Wake plot", duration_guess=5)
    def wake_plot(self, params, **kwargs):
        """
        Simulates and displays the wake effects of the wind farm, showing the distribution
        of wind speed and direction as well as the boundary of the wind farm. Also displays
        key metrics such as AEP (Annual Energy Production), wake loss, and the number of turbines.
        """
        # gather data
        points = convert_to_points(params.assemble.polygon.points)

        # wind farm model
        windfarm = get_wind_farm_model(points, params.visualize.turbine)

        # initiliaze turbine positions
        x, y = get_initial_turbine_positions(points, params.visualize.turbine, params.visualize.turbine_spacing)

        # simulation
        windfarm_simulated = windfarm(x, y, wd=params.visualize.wind_direction, ws=params.visualize.wind_speed)

        # define flow map and use a low resolution to decrease calculation times (but also accuracy)
        grid = py_wake.HorizontalGrid(x=None, y=None, resolution=50, extend=1.5)
        flow_map = windfarm_simulated.flow_map(grid=grid, wd=params.visualize.wind_direction, ws=params.visualize.wind_speed)

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
        png = vkt.File()
        fig.savefig(png.source, format="png", dpi=IMAGE_DPI)
        plt.close()

        # AEP, loss and number of turbines
        data = vkt.DataGroup(
            vkt.DataItem("AEP", value=calculate_aep(params), suffix="10^6 GWh"),
            vkt.DataItem("Loss", value=calculate_loss(params), suffix="%"),
            vkt.DataItem("Number of turbines", value=number_of_turbines(params)),
        )
        return vkt.ImageAndDataResult(png, data)

    @vkt.ImageAndDataView("Optimized positions", duration_guess=5)
    def optimized_positions(self, params, **kwargs):
        """
        Optimizes and displays the positions of the wind turbines within the site.
        Also provides the optimized Annual Energy Production (AEP) and its percentage increase
        from the initial configuration.
        """
        # optimize positions
        points = convert_to_points(params.assemble.polygon.points)
        _, aep, _, optimized_positions_png_s = optimize_turbine_positions(points, params.visualize.turbine, params.visualize.turbine_spacing, params.optimize.number_of_iterations)

        # save (increase of) AEP
        increase = (aep[-1] - aep[0]) / aep[0] * 100
        aep_data = {"aep": aep[-1] / 1e6, "increase": increase}

        # collect data group
        aep_data_group = vkt.DataGroup(
            vkt.DataItem(
                "AEP (optimal)",
                value=aep_data["aep"],
                suffix="10^6 GWh",
                number_of_decimals=2,
            ),
            vkt.DataItem(
                "AEP (increase)",
                value=aep_data["increase"],
                suffix="%",
                number_of_decimals=2,
            ),
        )

        optimized_positions_png = deserialize(optimized_positions_png_s)
        return vkt.ImageAndDataResult(optimized_positions_png, aep_data_group)

    def optimization_routine(self, params, **kwargs):
        """
        Executes the optimization routine for the wind turbine positions,
        tracking the computation time and the resulting Annual Energy Production (AEP).
        Returns an optimization result that includes a convergence info and plot.
        """
        # optimize positions
        points = convert_to_points(params.assemble.polygon.points)
        t, aep, convergence_png_s, _ = optimize_turbine_positions(points, params.visualize.turbine, params.visualize.turbine_spacing, params.optimize.number_of_iterations)

        # convergence result elements
        results = [vkt.OptimizationResultElement(params, {"time": round(_t - t[0], 2), "aep": round(_aep / 1e6, 3)}) for _t, _aep in zip(t, aep)]
        output_headers = {"time": "Computation time (s)", "aep": "AEP (10^6 GWh)"}

        # convergence plot
        convergence_png = deserialize(convergence_png_s)

        return vkt.OptimizationResult(results, output_headers=output_headers, image=vkt.ImageResult(convergence_png))
