"""Create Jupyter widget layout for Check_Warmup and Check_Rebin."""
# pylint: disable=invalid-name
import math

import ipywidgets as widgets
import matplotlib.pyplot as plt

def _create_fig(N, ncols=2):
    if N == 1:
        plt.ioff()
        fig, axes0 = plt.subplots(1, 1, constrained_layout=True)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_position = 'top'
        plt.ion()
        return fig, [axes0]
    # if ncols is None:
    #     ncols = math.ceil(math.sqrt(N))
    nrows = math.ceil(N/ncols)

    plt.ioff()
    fig, axes0 = plt.subplots(
        nrows, ncols,
        sharex='all',
        constrained_layout=True,
    )
    fig.canvas.header_visible = False
    fig.canvas.toolbar_position = 'top'
    # fig.canvas.footer_visible = False
    plt.ion()
    return fig, list(axes0.flat)

def init_layout(names, ncols=3, n_plots=5, int_names=('N_max:', 'N_skip:')):
    """Create Jupyter widget layout for Check_Warmup and Check_Rebin."""
    # pylint: disable=too-many-locals
    select = widgets.Select(
        options=names,
        rows=3,
        layout=widgets.Layout(width='99.5%'),
    )

    int_texts = [
        widgets.BoundedIntText(
        value=7,
        min=0,
        max=10,
        step=1,
        description=name,
        disabled=False,
        continuous_update=False,
        layout=widgets.Layout(width='170px'),
        ) for name in int_names]

    button_next = widgets.Button(
        description='Next',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Next go to next simulation',
        icon='', # (FontAwesome names without the `fa-` prefix)
        layout=widgets.Layout(width='100px'),
    )

    button_clear = widgets.Button(
        description='Clear log',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        # tooltip='Next',
        icon='', # (FontAwesome names without the `fa-` prefix)
        layout=widgets.Layout(width='100px'),
    )

    log = widgets.Output()

    def button_next_clicked(b):
        # pylint: disable=no-member
        del b
        with log:
            if len(select.options) > select.index+1:
                select.index += 1
            else:
                print('At the end')
    button_next.on_click(button_next_clicked)

    def button_clear_clicked(b):
        del b
        log.clear_output()
    button_clear.on_click(button_clear_clicked)

    controls = widgets.VBox([
        select,
        widgets.HBox([button_next, *int_texts, button_clear])
    ])

    fig, axs = _create_fig(n_plots, ncols=ncols)

    accordion = widgets.Accordion(children=[log])
    accordion.set_title(0, 'Log')
    gui = widgets.VBox([fig.canvas, controls, accordion])
    return (gui, log, axs, *int_texts, select)
