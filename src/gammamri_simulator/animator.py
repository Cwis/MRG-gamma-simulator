import matplotlib
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from mpl_toolkits.mplot3d import proj3d

from gammamri_simulator import simulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


def get_text(config):
    """Get opacity and display text pulseSeq event text flashes in 3D plot and store in config.

    :param config: configuration dictionary.
    :return:
    """

    # Setup display text related to pulseSeq events:
    config["RFtext"] = np.full([len(config["t"])], "", dtype=object)
    config["Gtext"] = np.full([len(config["t"])], "", dtype=object)
    config["spoiltext"] = "spoiler"
    config["RFalpha"] = np.zeros([len(config["t"])])
    config["Galpha"] = np.zeros([len(config["t"])])
    config["spoilAlpha"] = np.zeros([len(config["t"])])

    for rep in range(config["nTR"]):
        start_frame = rep * config["nFramesPerTR"]
        for i, event in enumerate(config["events"]):
            first_frame, last_frame = simulator.get_event_frames(config, i)
            first_frame += start_frame
            last_frame += start_frame

            if "RFtext" in event:
                config["RFtext"][first_frame:] = event["RFtext"]
                config["RFalpha"][first_frame : last_frame + 1] = 1.0
            if any(
                "{}text".format(g) in event for g in ["Gx", "Gy", "Gz"]
            ):  # gradient event
                Gtext = ""
                for g in ["Gx", "Gy", "Gz"]:
                    if "{}text".format(g) in event:
                        Gtext += "  " + event["{}text".format(g)]
                config["Gtext"][first_frame:] = Gtext
                config["Galpha"][first_frame : last_frame + 1] = 1.0
            if "spoil" in event and event["spoil"]:
                config["spoilAlpha"][first_frame] = 1.0


def resample_time_frames(vectors, config, animation: bool = False):
    """Resample (interpolate) given vectors corresponding to time vector config['t'] on time vector config['tFrames].
    Also resample text and alpha channels in config similiarly.

    :param vectors:
    :param config:
    :param animation: specify animation mode
    :return:
    """
    config["tFrames"] = simulator.get_prescribed_time_vector(
        config, config["nTR"], animation
    )
    new_shape = list(vectors.shape)
    new_shape[6] = len(config["tFrames"])
    resampled_vectors = np.zeros(new_shape)
    for x in range(new_shape[0]):
        for y in range(new_shape[1]):
            for z in range(new_shape[2]):
                for c in range(new_shape[3]):
                    for i in range(new_shape[4]):
                        for dim in range(new_shape[5]):
                            resampled_vectors[x, y, z, c, i, dim, :] = np.interp(
                                config["tFrames"],
                                config["t"],
                                vectors[x, y, z, c, i, dim, :],
                            )

    # resample text alpha channels:
    for channel in ["RFalpha", "Galpha", "spoilAlpha"]:
        alpha_vector = np.zeros([len(config["tFrames"])])
        for i in range(len(alpha_vector)):
            if i == len(alpha_vector) - 1:
                ks = np.where(config["t"] >= config["tFrames"][i])[0]
            else:
                ks = np.where(
                    np.logical_and(
                        config["t"] >= config["tFrames"][i],
                        config["t"] < config["tFrames"][i + 1],
                    )
                )[0]
            alpha_vector[i] = np.max(config[channel][ks])
        config[channel] = alpha_vector

    # resample text:
    for text in ["RFtext", "Gtext"]:
        text_vector = np.full([len(config["tFrames"])], "", dtype=object)
        for i in range(len(text_vector)):
            k = np.where(config["t"] >= config["tFrames"][i])[0][0]
            text_vector[i] = config[text][k]
        config[text] = text_vector

    return resampled_vectors


def fade_text_flashes(config, fade_time=1.0):
    """Modify text alpha channels such that the text flashes fade

    :param config: configuration dictionary.
    :param fade_time: time of fade in seconds
    :return:
    """
    decay = 1.0 / (config["fps"] * fade_time)  # alpha decrease per frame
    for channel in ["RFalpha", "Galpha", "spoilAlpha"]:
        for i in range(1, len(config[channel])):
            if config[channel][i] == 0:
                config[channel][i] = max(0, config[channel][i - 1] - decay)


colors = {
    "bg": [1, 1, 1],
    "circle": [0, 0, 0, 0.03],
    "axis": [0.5, 0.5, 0.5],
    "text": [0.05, 0.05, 0.05],
    "spoilText": [0.5, 0, 0],
    "RFtext": [0, 0.5, 0],
    "Gtext": [80 / 256, 80 / 256, 0],
    "comps": [
        [0.3, 0.5, 0.2],
        [0.1, 0.4, 0.5],
        [0.5, 0.3, 0.2],
        [0.5, 0.4, 0.1],
        [0.4, 0.1, 0.5],
        [0.6, 0.1, 0.3],
    ],
    "boards": {
        "w1": [0.5, 0, 0],
        "Gx": [0, 0.5, 0],
        "Gy": [0, 0.5, 0],
        "Gz": [0, 0.5, 0],
    },
    "kSpacePos": [1, 0.5, 0],
}


# Matplotlib >= 3.5 => extend Arrow3d cause missing do_3d_projection
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        # xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M) # deprecated
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# End fix


def plot_frame_3D(config, vectors, frame, output):
    """Creates a plot of magnetization vectors in a 3D view.

    Args:
        config: configuration dictionary.
        vectors:    numpy array of size [nx, ny, nz, nComps, nIsochromats, 3, nFrames].
        frame:  which frame to plot.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    """
    nx, ny, nz, nComps, nIsoc = vectors.shape[:5]

    # Create 3D axes
    if nx * ny * nz == 1 or config["collapseLocations"]:
        aspect = 0.952  # figure aspect ratio
    elif nz == 1 and ny == 1 and nx > 1:
        aspect = 0.6
    elif nz == 1 and nx > 1 and ny > 1:
        aspect = 0.75
    else:
        aspect = 1
    fig_size = 5  # figure size in inches
    canvas_width = fig_size
    canvas_height = fig_size * aspect
    fig = plt.figure(figsize=(canvas_width, canvas_height), dpi=output["dpi"])
    ax_limit = max(nx, ny, nz) / 2 + 0.5
    if config["collapseLocations"]:
        ax_limit = 1.0
    # ax = fig.gca( # deprecated matplotlib 3.4
    #     projection="3d",
    #     xlim=(-ax_limit, ax_limit),
    #     ylim=(-ax_limit, ax_limit),
    #     zlim=(-ax_limit, ax_limit),
    #     fc=colors["bg"],
    # )
    fig.add_subplot(projection="3d", fc=colors["bg"])
    ax = fig.gca()
    ax.set_xlim(-ax_limit, ax_limit)
    ax.set_ylim(-ax_limit, ax_limit)
    ax.set_zlim(-ax_limit, ax_limit)

    if nx * ny * nz > 1 and not config["collapseLocations"]:
        azim = -78  # azimuthal angle of x-y-plane
        ax.view_init(azim=azim)  # ax.view_init(azim=azim, elev=elev)
    ax.set_axis_off()
    width = 1.65  # to get tight cropping
    height = width / aspect
    left = (1 - width) / 2
    bottom = (1 - height) / 2
    if nx * ny * nz == 1 or config["collapseLocations"]:  # shift to fit legend
        left += 0.035
        bottom += -0.075
    else:
        bottom += -0.085
    ax.set_position([left, bottom, width, height])

    if output["drawAxes"]:
        # Draw axes circles
        for i in ["x", "y", "z"]:
            circle = Circle((0, 0), 1, fill=True, lw=1, fc=colors["circle"])
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

        # Draw x, y, and z axes
        ax.plot([-1, 1], [0, 0], [0, 0], c=colors["axis"], zorder=-1)  # x-axis
        ax.text(
            1.08,
            0,
            0,
            r"$x^\prime$",
            horizontalalignment="center",
            color=colors["text"],
        )
        ax.plot([0, 0], [-1, 1], [0, 0], c=colors["axis"], zorder=-1)  # y-axis
        ax.text(
            0,
            1.12,
            0,
            r"$y^\prime$",
            horizontalalignment="center",
            color=colors["text"],
        )
        ax.plot([0, 0], [0, 0], [-1, 1], c=colors["axis"], zorder=-1)  # z-axis
        ax.text(0, 0, 1.05, r"$z$", horizontalalignment="center", color=colors["text"])

    # Draw title:
    fig.text(
        0.5,
        1,
        config["title"],
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        color=colors["text"],
    )

    # Draw time
    time = config["tFrames"][frame % (len(config["t"]) - 1)]  # frame time [msec]
    time_text = fig.text(
        0,
        0,
        "time = %.1f msec" % (time),
        color=colors["text"],
        verticalalignment="bottom",
    )

    # TODO: put isochromats in this order from start
    order = [int((nIsoc - 1) / 2 - abs(m - (nIsoc - 1) / 2)) for m in range(nIsoc)]
    thres = 0.075 * ax_limit  # threshold on vector magnitude for shrinking
    if "rotate" in output:
        rot_freq = (
            output["rotate"] * 1e-3
        )  # coordinate system rotation relative resonance frequency [kHz]
        rot_mat = simulator.get_rotation_matrix(
            2 * np.pi * rot_freq * time, 2
        )  # rotation matrix for rotating coordinate system

    pos = [0, 0, 0]

    # Draw magnetization vectors
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nComps):
                    for m in range(nIsoc):
                        col = colors["comps"][(c) % len(colors["comps"])]
                        M = vectors[x, y, z, c, m, :3, frame]
                        if not config["collapseLocations"]:
                            pos = (
                                vectors[x, y, z, c, m, 3:, frame] / config["locSpacing"]
                            )
                        if "rotate" in output:
                            M = np.dot(
                                M, rot_mat
                            )  # rotate vector relative to coordinate system
                        Mnorm = np.linalg.norm(M)
                        alpha = 1.0 - 2 * np.abs((m + 0.5) / nIsoc - 0.5)
                        if Mnorm > thres:
                            arrowScale = 20
                        else:
                            arrowScale = (
                                20 * Mnorm / thres
                            )  # Shrink arrowhead close to origo
                        ax.add_artist(
                            Arrow3D(
                                [pos[0], pos[0] + M[0]],
                                [-pos[1], -pos[1] + M[1]],
                                [-pos[2], -pos[2] + M[2]],
                                mutation_scale=arrowScale,
                                arrowstyle="-|>",
                                shrinkA=0,
                                shrinkB=0,
                                lw=2,
                                color=col,
                                alpha=alpha,
                                zorder=order[m] + nIsoc * int(100 * (1 - Mnorm)),
                            )
                        )

    # Draw "spoiler" and "FA-pulse" text
    fig.text(
        1,
        0.94,
        config["RFtext"][frame],
        fontsize=14,
        alpha=config["RFalpha"][frame],
        color=colors["RFtext"],
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.text(
        1,
        0.88,
        config["Gtext"][frame],
        fontsize=14,
        alpha=config["Galpha"][frame],
        color=colors["Gtext"],
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.text(
        1,
        0.82,
        config["spoiltext"],
        fontsize=14,
        alpha=config["spoilAlpha"][frame],
        color=colors["spoilText"],
        horizontalalignment="right",
        verticalalignment="top",
    )

    # Draw legend:
    for c in range(nComps):
        col = colors["comps"][(c) % len(colors["comps"])]
        ax.plot(
            [0, 0],
            [0, 0],
            [0, 0],
            "-",
            lw=2,
            color=col,
            alpha=1.0,
            label=config["components"][c]["name"],
        )
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        [
            plt.Line2D(
                (0, 1), (0, 0), lw=2, color=colors["comps"][(c) % len(colors["comps"])]
            )
            for c, handle in enumerate(handles)
        ],
        labels,
        loc=2,
        bbox_to_anchor=[-0.025, 0.94],
    )
    leg.draw_frame(False)
    for text in leg.get_texts():
        text.set_color(colors["text"])

    return fig


def plot_frame_mt(config, signal, frame, output):
    """Creates a plot of transversal or longituinal magnetization over time.

    Args:
        config: configuration dictionary.
        signal: numpy array of size [nComps, 3, nFrames].
        frame:  which frame to plot up to.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    """
    if output["type"] not in ["xy", "z"]:
        raise Exception(
            'output "type" must be 3D, kspace, psd, xy (transversal) or z (longitudinal)'
        )

    # create diagram
    xmin, xmax = output["tRange"]

    if output["type"] == "xy":
        if "abs" in output and not output["abs"]:
            ymin, ymax = -1, 1
        else:
            ymin, ymax = 0, 1
    elif output["type"] == "z":
        ymin, ymax = -1, 1
    fig = plt.figure(figsize=(5, 2.7), facecolor=colors["bg"], dpi=output["dpi"])
    # ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors["bg"]) # deprecated
    fig.add_subplot(fc=colors["bg"])
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)  # remove default axes
    ax.grid()
    plt.title(config["title"], color=colors["text"])
    plt.xlabel("time[ms]", horizontalalignment="right", color=colors["text"])
    if output["type"] == "xy":
        if "abs" in output and not output["abs"]:
            ax.xaxis.set_label_coords(1.1, 0.475)
            plt.ylabel("$M_x, M_y$", rotation=0, color=colors["text"])
        else:  # absolute value of transversal magnetization
            ax.xaxis.set_label_coords(1.1, 0.1)
            plt.ylabel("$|M_{xy}|$", rotation=0, color=colors["text"])
    elif output["type"] == "z":
        ax.xaxis.set_label_coords(1.1, 0.475)
        plt.ylabel("$M_z$", rotation=0, color=colors["text"])
    ax.yaxis.set_label_coords(-0.07, 0.95)
    plt.tick_params(axis="y", labelleft="off")
    plt.tick_params(axis="x", colors=colors["text"])
    ax.xaxis.set_ticks_position("none")  # tick markers
    ax.yaxis.set_ticks_position("none")

    # draw x and y axes as arrows
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height  # get width and height of axes object
    hw = 1 / 25 * (ymax - ymin)  # manual arrowhead width and length
    hl = 1 / 25 * (xmax - xmin)
    yhw = (
        hw / (ymax - ymin) * (xmax - xmin) * height / width
    )  # compute matching arrowhead length and width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height
    ax.arrow(
        xmin,
        0,
        (xmax - xmin) * 1.05,
        0,
        fc=colors["text"],
        ec=colors["text"],
        lw=1,
        head_width=hw,
        head_length=hl,
        clip_on=False,
        zorder=100,
    )
    ax.arrow(
        xmin,
        ymin,
        0,
        (ymax - ymin) * 1.05,
        fc=colors["text"],
        ec=colors["text"],
        lw=1,
        head_width=yhw,
        head_length=yhl,
        clip_on=False,
        zorder=100,
    )

    # Draw magnetization vectors
    nComps = signal.shape[0]
    if output["type"] == "xy":
        for c in range(nComps):
            col = colors["comps"][c % len(colors["comps"])]
            if (
                "abs" in output and not output["abs"]
            ):  # real and imag part of transversal magnetization
                ax.plot(
                    config["tFrames"][: frame + 1],
                    signal[c, 0, : frame + 1],
                    "-",
                    lw=2,
                    color=col,
                )
                col = colors["comps"][c + nComps + 1 % len(colors["comps"])]
                ax.plot(
                    config["tFrames"][: frame + 1],
                    signal[c, 1, : frame + 1],
                    "-",
                    lw=2,
                    color=col,
                )
            else:  # absolute value of transversal magnetization
                ax.plot(
                    config["tFrames"][: frame + 1],
                    np.linalg.norm(signal[c, :2, : frame + 1], axis=0),
                    "-",
                    lw=2,
                    color=col,
                )
        # plot sum component if both water and fat (special case)
        if all(
            key in [comp["name"] for comp in config["components"]]
            for key in ["water", "fat"]
        ):
            col = colors["comps"][nComps % len(colors["comps"])]
            if (
                "abs" in output and not output["abs"]
            ):  # real and imag part of transversal magnetization
                ax.plot(
                    config["tFrames"][: frame + 1],
                    np.mean(signal[:, 0, : frame + 1], 0),
                    "-",
                    lw=2,
                    color=col,
                )
                col = colors["comps"][2 * nComps + 1 % len(colors["comps"])]
                ax.plot(
                    config["tFrames"][: frame + 1],
                    np.mean(signal[:, 1, : frame + 1], 0),
                    "-",
                    lw=2,
                    color=col,
                )
            else:  # absolute value of transversal magnetization
                ax.plot(
                    config["tFrames"][: frame + 1],
                    np.linalg.norm(np.mean(signal[:, :2, : frame + 1], 0), axis=0),
                    "-",
                    lw=2,
                    color=col,
                )

    elif output["type"] == "z":
        for c in range(nComps):
            col = colors["comps"][(c) % len(colors["comps"])]
            ax.plot(
                config["tFrames"][: frame + 1],
                signal[c, 2, : frame + 1],
                "-",
                lw=2,
                color=col,
            )

    return fig


def plot_frame_kspace(config, frame, output):
    """Creates a plot of k-space position for the given frame.

    Args:
        config: configuration dictionary.
        frame:  which frame to plot.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    """

    gyro = float(config["gyro"])

    # TODO: support for 3D k-space
    kmax = 1 / (2 * config["locSpacing"])
    xmin, xmax = -kmax, kmax
    ymin, ymax = -kmax, kmax
    fig = plt.figure(figsize=(5, 5), facecolor=colors["bg"], dpi=output["dpi"])
    # ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors["bg"]) # deprecated
    fig.add_subplot(fc=colors["bg"])
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_color(colors["text"])
    ax.grid()
    plt.title(config["title"], color=colors["text"])
    plt.xlabel("$k_x$ [m$^{-1}$]", horizontalalignment="right", color=colors["text"])
    plt.ylabel("$k_y$ [m$^{-1}$]", rotation=0, color=colors["text"])
    plt.tick_params(axis="y", colors=colors["text"])
    plt.tick_params(axis="x", colors=colors["text"])

    frame_time = config["tFrames"][frame] % config["TR"]
    kx, ky, kz = 0, 0, 0
    for i, event in enumerate(config["events"]):
        first_frame, last_frame = simulator.get_event_frames(config, i)
        if event["t"] < frame_time:
            dur = min(frame_time, config["t"][last_frame]) - config["t"][first_frame]
            if "spoil" in event and event["spoil"]:
                kx, ky, kz = 0, 0, 0
            kx += gyro * event["Gx"] * dur / 1e3
            ky += gyro * event["Gy"] * dur / 1e3
            kz += gyro * event["Gz"] * dur / 1e3
        else:
            break
    ax.plot(kx, ky, ".", markersize=10, color=colors["kSpacePos"])
    return fig


def plot_frame_psd(config, frame, output):
    """Creates a plot of the pulse sequence diagram.

    Args:
        config: configuration dictionary.
        frame:  which frame to indicate by vertical line.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    """
    if "fig" in output:
        fig, timeLine = output["fig"]
    else:
        xmin, xmax = output["tRange"]
        ymin, ymax = 0, 5
        fig = plt.figure(figsize=(5, 5), facecolor=colors["bg"], dpi=output["dpi"])
        # ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors["bg"]) # deprecated
        fig.add_subplot(fc=colors["bg"])
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        for side in ["bottom", "right", "top", "left"]:
            ax.spines[side].set_visible(False)  # remove default axes
        plt.title(config["title"], color=colors["text"])
        plt.xlabel("time[ms]", horizontalalignment="right", color=colors["text"])
        plt.tick_params(axis="y", labelleft="off")
        plt.tick_params(axis="x", colors=colors["text"])
        ax.xaxis.set_ticks_position("none")  # tick markers
        ax.yaxis.set_ticks_position("none")

        # draw x axis as arrow
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height  # get width and height of axes object
        hw = 1 / 25 * (ymax - ymin)  # manual arrowhead width and length
        hl = 1 / 25 * (xmax - xmin)
        yhw = (
            hw / (ymax - ymin) * (xmax - xmin) * height / width
        )  # compute matching arrowhead length and width
        yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height
        ax.arrow(
            xmin,
            0,
            (xmax - xmin) * 1.05,
            0,
            fc=colors["text"],
            ec=colors["text"],
            lw=1,
            head_width=hw,
            head_length=hl,
            clip_on=False,
            zorder=100,
        )

        boards = {
            "w1": {"ypos": 4},
            "Gx": {"ypos": 3},
            "Gy": {"ypos": 2},
            "Gz": {"ypos": 1},
        }
        for board in boards:
            boards[board]["signal"] = [0]
        t = [0]
        for event in config["events"]:
            for board in boards:
                boards[board]["signal"].append(
                    boards[board]["signal"][-1]
                )  # end of previous event:
                boards[board]["signal"].append(event[board])  # start of this event:
            t.append(event["t"])  # end of previous event:
            t.append(event["t"])  # start of this event:

        boards["w1"]["scale"] = 0.48 / np.max(
            [np.abs(w) for w in boards["w1"]["signal"] if np.abs(w) < 50]
        )
        if "gmax" not in output:
            output["gmax"] = np.max(
                np.abs(
                    np.concatenate(
                        (
                            boards["Gx"]["signal"],
                            boards["Gy"]["signal"],
                            boards["Gz"]["signal"],
                        )
                    )
                )
            )
        boards["Gx"]["scale"] = boards["Gy"]["scale"] = boards["Gz"]["scale"] = (
            0.48 / output["gmax"]
        )

        for board in ["w1", "Gx", "Gy", "Gz"]:
            ax.plot(
                t,
                boards[board]["ypos"]
                + np.array(boards[board]["signal"]) * boards[board]["scale"],
                lw=1,
                color=colors["boards"][board],
            )
            ax.plot(
                [xmin, xmax],
                [boards[board]["ypos"], boards[board]["ypos"]],
                color=colors["text"],
                lw=1,
                clip_on=False,
                zorder=100,
            )
            ax.text(
                0,
                boards[board]["ypos"],
                board,
                fontsize=14,
                color=colors["text"],
                horizontalalignment="right",
                verticalalignment="center",
            )

        # plot vertical time line:
        (timeLine,) = ax.plot(
            [
                config["tFrames"][frame] % config["TR"],
                config["tFrames"][frame] % config["TR"],
            ],
            [0, 5],
            color=colors["text"],
            lw=1,
            clip_on=False,
            zorder=100,
        )
        output["fig"] = fig, timeLine
    timeLine.set_xdata(
        [
            config["tFrames"][frame] % config["TR"],
            config["tFrames"][frame] % config["TR"],
        ]
    )
    fig.canvas.draw()
    return fig
