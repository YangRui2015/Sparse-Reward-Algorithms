import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean
- If included, actor and critic neural network parameters are reset

2. Testing boolean
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks. 
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean
- If included, training will be visualized

4. Train Only boolean
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean
- If included, summary of each transition will be printed
"""


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threadings",
        type=int,
        help="Number of threadings running one FLAGS",
        default=1,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        help='Total episodes to run',
        default=5000
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        help="How often to save model",
        default=50,
    )

    parser.add_argument(
        "--layers",
        type=int,
        help="How many layer of the HRL",
        default=2,
    )

    parser.add_argument(
        "--her",
        action='store_true',
        help='if use her or not',
    )

    parser.add_argument(
        '--rnd',
        action='store_true',
        help='if use rnd or not',
    )

    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     help="random seed",
    #     default=5
    # )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='normalize observation and goal'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--env',
        type=str,
        default="reach",
        help='The environment to run'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.threadings > 1:
        FLAGS.show = False

    if FLAGS.rnd:
        FLAGS.normalize = True

    print(FLAGS)
    import pdb
    pdb.set_trace()


    return FLAGS
