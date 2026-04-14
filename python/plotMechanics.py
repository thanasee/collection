#!/usr/bin/env python

from sys import argv, exit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set Times New Roman (with fallback)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # for math consistency

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange']

PROPERTY_MAP = {
    "1": {"data_type": "Young's Modulus",  "unit": "N/m", "decimal": 0, "save_file": "Young.png"},
    "2": {"data_type": "Poisson's Ratio",  "unit": "",    "decimal": 2, "save_file": "Poisson.png"},
    "3": {"data_type": "Shear Modulus",    "unit": "N/m", "decimal": 0, "save_file": "Shear.png"},
}

def usage():
    
    text = """
Usage: plotMechanics <input file> [input file2 ... input file6]

This script plot Young's modulus and Poisson's Ratio as functions of crystal orientation.
Title can choose by:
1 - Young's Modulus
2 - Poisson's Ratio
3 - Shear Modulus

This script was inspired by Klichchupong Dabsamut
and developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def ask_property():
    
    print("""Enter type of your data
1) Young's Modulus
2) Poisson's Ratio
3) Shear Modulus""")
    while True:
        choice = input()
        if choice in PROPERTY_MAP:
            meta = PROPERTY_MAP[choice]
            data_type = meta["data_type"]
            unit = meta["unit"]
            title = f"{data_type} ({unit})" if unit else data_type
            return {**meta, "title": title}
        print("Choose again!")

def ask_material_label(index):
    
    while True:
        label = input(f"Enter name of {index} material: ")
        if label:
            return label

def ask_positive_float(prompt):
    
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("Must be a positive number.")
        except ValueError:
            print("Invalid input. Enter a valid number.")

def ask_step(ymax):
    
    factors = get_factors(ymax)
    print(f"Suggested step sizes (factors of {int(ymax)}): {factors}")
    while True:
        try:
            step = float(input("Enter your step size: "))
            if 0 < step < ymax:
                return step
            print(f"Step must be positive and less than {ymax}.")
        except ValueError:
            print("Invalid input. Enter a valid number.")

def load_data(filepath):
    
    with open(filepath, 'r') as f:
        lines = np.array(
            [line.split() for line in f
             if line.strip() and not line.strip().startswith('#')],
            dtype=float
        )
    degrees = lines[:, 0]
    radians = np.radians(degrees)
    values  = lines[:, 1]
    
    return degrees, radians, values

def get_factors(n):
    
    n = int(round(n))
    factors = sorted(set(
        f for i in range(1, int(n**0.5) + 1) if n % i == 0
        for f in (i, n // i)
    ))
    
    return factors

def build_tick_labels(ymax, step, decimal):
    
    number = int(np.floor(ymax / step))
    tick = np.arange(1, number + 1) * step
    labels = np.round(np.concatenate((-tick[::-1], [0.0], tick)), decimals=decimal)
    
    return labels

def setup_figure():
    
    fig = plt.figure(dpi=300)
    ax  = fig.add_subplot(111)
    axp = fig.add_subplot(111, polar=True)
    
    return fig, ax, axp

def plot_material(axp, radians, values, color, label, neg_envelope_labeled):
    
    if (values < 0.0).any():
        neg_label = "|value| envelope" if not neg_envelope_labeled else "_nolegend_"
        axp.plot(radians, np.abs(values), linestyle='dashed', linewidth=1.5,
                 color=color, label=neg_label)
        neg_envelope_labeled = True
 
    axp.plot(radians, values, linestyle='solid', linewidth=2, color=color, label=label)
    
    return neg_envelope_labeled

def configure_cartesian_axis(ax, sub_y_labels, ymax, title, decimal):
    
    formatted = [f"{abs(y):.{decimal}f}" for y in sub_y_labels]
    ax.set_ylim(-ymax, ymax)
    ax.set_yticks(sub_y_labels)
    ax.set_yticklabels(formatted, fontsize=12)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_xticklabels([])
    ax.set_ylabel(title, fontsize=18)

def configure_polar_axis(axp, sub_y_labels, ymax):
    
    sub_degree_labels = np.arange(0, 360, 30)
    sub_r_labels = np.delete(sub_y_labels, np.where(sub_y_labels <= 0))

    axp.set_xticks(np.radians(sub_degree_labels))
    axp.set_xticklabels(
        [str(d) + '\u00B0' for d in sub_degree_labels],
        fontsize=12, color='black'
    )
    axp.patch.set_alpha(0)
    axp.set_rlim(0, ymax)
    axp.set_rticks(sub_r_labels)
    axp.set_yticklabels([])
    axp.set_title('')
    axp.grid(True)

    axp.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    leg = axp.get_legend()
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.5)

def main():
    
    if '-h' in argv or len(argv) < 2 or len(argv) > 7:
        usage()
    
    input_files = argv[1:]
    meta = ask_property()

    data_type = meta["data_type"]
    unit = meta["unit"]
    decimal = meta["decimal"]
    title = meta["title"]
    save_file = meta["save_file"]

    fig, ax, axp = setup_figure()

    neg_envelope_labeled = False
    for i, filepath in enumerate(input_files):
        degrees, radians, values = load_data(filepath)
        label = ask_material_label(i + 1)

        idx_largest = np.argmax(np.abs(values))
        print(f"Your largest {data_type} is {values[idx_largest]:>6.{decimal}f} {unit} "
              f"at {degrees[idx_largest]:>5.1f}\u00B0")

        color = COLORS[i % len(COLORS)]
        neg_envelope_labeled = plot_material(axp, radians, values, color, label,
                                             neg_envelope_labeled)

    ymax = ask_positive_float("Enter your maximum size: ")
    step = ask_step(ymax)

    sub_y_labels = build_tick_labels(ymax, step, decimal)
    configure_cartesian_axis(ax, sub_y_labels, ymax, title, decimal)
    configure_polar_axis(axp, sub_y_labels, ymax)

    plt.savefig(save_file, dpi=300, bbox_inches='tight', format='png')

if __name__ == '__main__':
    main()
