import matplotlib.pyplot as plt
from pylab import *
import tensorflow as tf
import numpy as np
import random
import time


def set_global_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)


def print_summary(FLAGS,env):

    print("\n- - - - - - - - - - -")
    print("Task Summary: ","\n")
    print("Environment: ", env.model_path)
    print("Total Episodes:", FLAGS.episodes)
    print("Hindsight Experiment Replay: ", FLAGS.her)
    print("Number of Layers: ", FLAGS.layers)
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("- - - - - - - - - - -", "\n\n")


def load_pickle_file(path="performance_log.p"):
    import pickle
    pkl_file = open(path, 'rb')
    f = pickle.load(pkl_file)
    pkl_file.close()
    print(f)
    return f


def clear_perfomance_data(path='performance_data.txt'):
    with open(path, 'w') as file:
        file.write('clear since :{}\n\n'.format(time.ctime()))
    print("clear performance finished.")


# 保存实验参数和结果
def save_performance(performance_list, FLAGS, thread_num=0):
    with open("performance_data.txt", "a+") as file:
        file.writelines("time: {}  , thread: {}\n".format(time.ctime(), thread_num))
        file.writelines("FLAGS: {} \n".format(str(FLAGS)))
        info = [str(x) + " " for x in performance_list]
        file.writelines(info)
        file.writelines("\n\n")
        print("thread {} save performance finished".format(thread_num))


def save_plot_figure(performance_list):
    plot(performance_list)
    plt.title("Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("performance.jpg")


def plot_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line[0] in "0123456789":
                num_list = [float(x) for x in line.split()]
                plot(num_list)
                plt.title("Performance")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.savefig(path.replace(".txt", ".jpg"))

            else:
                continue



if __name__=="__main__":
    clear_perfomance_data()