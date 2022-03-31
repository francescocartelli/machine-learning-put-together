from examples.htru2.data_utils import *
from plotting.plotting import *

output_dir = "./outputs/"

# Example of bayes error plot confrontation
if __name__ == "__main__":
    LTE = np.load(f"{output_dir}labels.npy")

    models = ["MVG", "NBG"]
    llr_list = [np.load(f"{output_dir}{model}_llr.npy") for model in models]

    plot_roc_curve(llr_list[0], LTE, 1000)

    #plot_multiple_bayes_error(llr_list, LTE, np.linspace(-3, 3, 21), legend=models)
