import pure_pursuit as pp
import numpy as np

col_seperator = " & "
row_seperator = " \\\\\n "

def pursuer_options(max_pursuers):
    return range(2, max_pursuers+1)

def mu_lod_cap_table(mus, max_n):
    table = np.zeros((len(mus), max_n-1))

    for i, mu in enumerate(mus):
        for j, n in enumerate(pursuer_options(max_n)):
            lod = pp.polygon_formation_capture_ratio_d(mu, n)
            table[i, j] = lod

    return table

def mu_lod_cap_header(max_n):
    return col_seperator.join(map(str, pursuer_options(max_n)))

def format_table_as_latex(table):
    """
    `table` is a 2d numpy array
    """
    return row_seperator.join(
        col_seperator.join(map("{:.2f}".format, row))
        for row in table
    )

if __name__ == "__main__":
    mus = [1.1, 1.5, 2, 2.5, 3, 4]
    max_ns = 10
    table = format_table_as_latex(mu_lod_cap_table(mus, max_ns))
    print(mu_lod_cap_header(max_ns) + row_seperator, end="")
    print(r"\hline")
    print(table)

    print(mus)
