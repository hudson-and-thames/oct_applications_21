import matplotlib.pyplot as plt


def plot_points_bands(idx, pos, pos_std, acc, acc_std, name):
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(10, 3), constrained_layout=True)

    ax1.plot(idx, pos, marker="", linestyle="-", label="pos", color="b")
    ax1.plot(idx, pos - 2 * pos_std, linestyle="-", color="b")
    ax1.plot(idx, pos + 2 * pos_std, linestyle="-", color="b")

    ax2 = ax1.twinx()
    ax2.plot(idx, acc, marker="", linestyle="-", label="acc", color="r")
    ax2.plot(idx, acc - 2 * acc_std, linestyle="-", color="r")
    ax2.plot(idx, acc + 2 * acc_std, linestyle="-", color="r")

    plt.legend(loc="lower left", prop={"size": 8})
    plt.title(name)
    return plt


def plot_simple(idx, data, name):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(idx, data, marker="", linestyle="-")
    plt.title(name)
    return plt


def plot_two_lines(idx, line1, line2, name):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(idx, line1, marker="", linestyle="-", color="b")
    ax.plot(idx, line2, marker="", linestyle="-", color="r")
    plt.title(name)
    return plt


def plot_two_lines_two_axis(idx, line1, line2, name):
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax1.plot(idx, line1, marker="", linestyle="-", color="b")
    ax2 = ax1.twinx()
    ax2.plot(idx, line2, marker="", linestyle="-", color="r")
    plt.title(name)
    return plt
