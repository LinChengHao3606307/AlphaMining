
import copy
import matplotlib.pyplot as plt
from yaml import warnings


class Train_data_holder:
    def __init__(self, all_categories: list[str],title:str = 'Training Loss Over Epochs'):
        self.title = title
        self.empty_epoch_data: dict[str, dict[str, float | int]] = {}
        for name in all_categories:
            self.empty_epoch_data[name] = {
                'sum_of_loss': 0.0,
                'num_of_frames': 0
            }
        self.record: list[dict[str, dict[str, float | int]]] = []
        self.fig = None
        self.ax = None
        self.lines = {}

    def start_new_epoch_record(self):
        ept_rec = copy.deepcopy(self.empty_epoch_data)
        self.record.append(ept_rec)

    def add_record_to_current_epoch(self, name: str, new_loss: float):
        self.record[-1][name]['sum_of_loss'] += new_loss
        self.record[-1][name]['num_of_frames'] += 1

    def get_last_epoch_avg_loss(self,category:str):
        if self.record[-1][category]['num_of_frames'] == 0:
            return float('inf')
        return self.record[-1][category]['sum_of_loss']/self.record[-1][category]['num_of_frames']

    def log_past_and_current(self):
        sep_str = '______'
        for _ in range(3):
            print()
        len_of_rec = len(self.record)
        for i in range(len_of_rec):
            rec_data_str = '     '
            if i == len_of_rec - 1:
                rec_data_str = ' cur '
            for key, value in self.record[i].items():
                loss_str = 'N/A'
                if value['num_of_frames'] != 0:
                    loss_str = f"{value['sum_of_loss'] / value['num_of_frames']:6f}"
                i_str = '[' + key + ': ' + loss_str + ']'
                rec_data_str += i_str + sep_str
            print(rec_data_str[:-len(sep_str)])

    def plot_graph(self):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()
            self.lines = {}
            plt.style.use('dark_background')
            plt.ion()
            plt.show(block=False)
            self.ax.set_xlim(left=0)
            self.ax.set_ylim(bottom=0)
            self.ax.grid(True, which='both', axis='both',
                         color='#4d4d4d',  # 深灰色，适配暗色背景
                         linestyle='--',  # 虚线样式
                         linewidth=0.8,  # 适当线宽
                         alpha=0.6)  # 半透明效果

            self.fig.patch.set_facecolor('black')
            self.ax.set_facecolor('black')

            c = (0, 1, 0)
            self.ax.tick_params(axis='x', colors=c)
            self.ax.tick_params(axis='y', colors=c)
            self.ax.spines['bottom'].set_color(c)
            self.ax.spines['bottom'].set_linewidth(3)
            self.ax.spines['left'].set_color(c)
            self.ax.spines['left'].set_linewidth(3)

        max_loss = 0
        total_epochs = len(self.record)
        self.ax.set_xlim(right=max(total_epochs, 1))  # Ensure minimum xlim of 1

        for category in self.empty_epoch_data.keys():
            valid_epochs = []
            avg_losses = []

            # Collect valid data points
            for ep_idx, epoch_data in enumerate(self.record):
                data = epoch_data[category]
                if data['num_of_frames'] != 0:
                    avg_loss = data['sum_of_loss'] / data['num_of_frames']
                    avg_losses.append(avg_loss)
                    valid_epochs.append(ep_idx)

            if not avg_losses:
                continue  # Skip categories with no data

            # Update plot data
            if category in self.lines:
                line = self.lines[category]
                line.set_data(valid_epochs, avg_losses)
            else:
                line, = self.ax.plot(valid_epochs, avg_losses, label=category)
                self.lines[category] = line

            # Track maximum loss for y-axis
            current_max = max(avg_losses)
            max_loss = max(max_loss, current_max)

        # Adjust axes limits
        if max_loss > 0:
            self.ax.set_ylim(top=max_loss * 1.25)
        else:
            self.ax.set_ylim(top=1)  # Default if no losses

        # Update plot elements
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.ax.legend()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Average Loss', color='white')
        self.ax.set_title(self.title, color='white')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def stop_showing(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.lines = {}