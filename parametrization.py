import viktor as vkt

from wind_farm import MAX_ITERATIONS, SITE_MAXIMUM_AREA, SITE_MINIMUM_AREA, TURBINES, WIND_BIN_NUMS, calculate_aep, calculate_loss, get_windfarm_area, number_of_turbines


class Parametrization(vkt.ViktorParametrization):
    assemble = vkt.Step("Select location", views=["site_location"], width=30)
    assemble.welcome_text = vkt.Text(
        f"""
# Welcome to wind farm modelling with PyWake! 💨
With this app you can design a wind farm in a few steps.

Start by selecting an area for your wind farm by drawing a polygon on the map.

In order to keep the current problem manageable, the area of your windfarm $A$ should satisfy
$$
{SITE_MINIMUM_AREA} < A < {SITE_MAXIMUM_AREA}
        """
        + r"""
\ ({\textrm{km}^2})
$$
Note that one of the biggest off-shore wind farms, known as the [Hornsea Wind Farm](https://en.wikipedia.org/wiki/Hornsea_Wind_Farm)
covers an area of almost 5000 $\textrm{km}^2$! Typically speaking wind farms are a couple hundreds of squared kilometers. Some 
examples include:
- [Walney Wind Farm](https://en.wikipedia.org/wiki/Walney_Wind_Farm) (~80 $\textrm{km}^2$);
- [Triton Knoll](https://en.wikipedia.org/wiki/Triton_Knoll) (~200 $\textrm{km}^2$);
- [Borssele Offshore Wind Farm](https://en.wikipedia.org/wiki/Borssele_Offshore_Wind_Farm) (~300 $\textrm{km}^2$).
"""
    )
    assemble.polygon = vkt.GeoPolygonField("")
    assemble.windfarm_area = vkt.OutputField("Wind farm area", suffix=r"$\textrm{km}^2$", value=get_windfarm_area, flex=50)

    _polygon_selected = vkt.IsNotNone(vkt.Lookup("assemble.polygon"))
    conditions = vkt.Step("Wind conditions", views="wind_rose", enabled=_polygon_selected)
    conditions.text = vkt.Text(
        """
# Wind conditions at your site
Wind conditions are gathered at your site location from the [Global Wind Atlas](https://globalwindatlas.info/en/)

Below you can edit the wind rose plot to your liking.
        """
    )
    conditions.number_wind_directions = vkt.OptionField("Number of wind direction bins", options=WIND_BIN_NUMS, default=WIND_BIN_NUMS[-1])
    conditions.number_wind_speeds = vkt.NumberField("Number of wind speed bins", variant="slider", min=2, max=4, default=4)

    visualize = vkt.Step("Visualize wakes", views="wake_plot", enabled=_polygon_selected)
    visualize.intro_text = vkt.Text(
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
    visualize.wake_effects_image = vkt.Image(path="wake_effects.jpg")
    visualize.revenue_text = vkt.Text(
        """

The goal is to maximize your wind farm's Annual Energy Production (AEP), 
while using a relatively low number of turbines. These values play an essential 
role in determining when you can expect a Return on Investment (ROI) of your wind farm. 
The output fields below as well the data menu in the wake plot (press the "<" on the right) 
show AEP, percentual loss and the number of turbines.
"""
    )
    visualize.aep = vkt.OutputField("AEP", value=calculate_aep, suffix=r"$\times 10 ^ 6 \ \textrm{GWh}$")
    visualize.loss = vkt.OutputField("Loss", value=calculate_loss, suffix="%")
    visualize.number_of_turbines = vkt.OutputField("Number of turbines", value=number_of_turbines)
    visualize.wind_velocity_text = vkt.Text(
        """
## Wind
Wind direction and -speed will change troughout the course of a year. However, you 
can get a good idea of what the typical wind conditions will be at your site by studying the wind rose
you generated in the 'Wind conditions' step. Try setting the wind direction and -speed to the most 
common values you gather from the wind rose. 
    """
    )
    visualize.wind_direction = vkt.NumberField("Wind direction", min=0, max=359, suffix="°", variant="slider", step=1, default=270)
    visualize.wind_speed = vkt.NumberField("Wind speed", min=4, max=30, suffix="m/s", variant="slider", step=0.1, default=10)
    visualize.wind_park_layout_text = vkt.Text(
        """
## Turbines 
Try to keep the follwing in mind, while designing your wind farm:
- Opting for a bigger turbine increases the energy production per individual turbine, 
but allows for a lower total number of turbines and increases wake effects;
- Additionally, turbines are typically spaced several multiples of their diameter away from each other. 
Increasing this spacing reduces wake effects as well as the total number of turbines and AEP. 
        """
    )
    visualize.turbine = vkt.OptionField("Type", options=TURBINES, default=TURBINES[0])
    visualize.turbine_spacing = vkt.NumberField("Spacing", default=13, min=13, max=18, suffix="turbine diameters", variant="slider", step=1) # coarse to reduce memory footprint

    optimize = vkt.Step( "Optimize turbine locations", views="optimized_positions", enabled=_polygon_selected)
    optimize.text = vkt.Text(
        """
# Optimize your wind farm

Many factors come into play when optmizing your wind farm's layout. The previous step
illustrates how this can complicate finding the most efficient and profitable wind farm.
Luckily, we can use the [Topfarm module](https://topfarm.pages.windenergy.dtu.dk/TopFarm2/) 
to automatically find optimal positions for the wind turbines. This 
optimization considers the wake effects resulting from the wind conditions at your chosen site.

By pressing the optimization button below you can further improve on your wind farm's layout.
After the routine has finished, you can update the "Optimized positions" view on the right to see 
the recommended changes!

Optimization can take anywhere from a few seconds to a few minutes, depending on the number of turbines
and iterations. Below you can alter the number of iterations to your liking
        """
    )
    optimize.positions = vkt.OptimizationButton("Optimize turbine positions", "optimization_routine", longpoll=True, flex=50)
    optimize.number_of_iterations = vkt.NumberField("Number of iterations", min=2, max=MAX_ITERATIONS, default=4, step=1, variant="slider", suffix=f"max. {MAX_ITERATIONS}", flex=50)
