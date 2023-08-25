import matplotlib.pyplot as plt
import os

class PlotSaver():
    def __init__(self, output_path, format):
        self.output_path = output_path
        self.format = format
        self.plots = []

        os.makedirs(self.output_path, exist_ok=True)

    def to_latex(self):
        figures = (
            f"""
\\begin{{figure}}[!htb]
\\includegraphics{{ {plot} }}
\\caption{{}}
\\end{{figure}}
            """
            for plot in self.plots
        )

        return "\n\n".join(figures)

    def save(self, name, latex=False, **kwargs):
        path = f"{self.output_path}/{name}.{self.format}"

        if latex:
            self.plots.append(path)

        plt.subplots_adjust(bottom=0.2)
        plt.savefig(
            path, 
            # bbox_extra_artists=(plt.gca().get_children()),
            **kwargs
        )
        plt.cla()
        plt.clf()
