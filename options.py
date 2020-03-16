import argparse

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
        default=10,
    )

    parser.add_argument(
        '--save_experience',
        action='store_true',
        help='if save experience for imitatation learning'
    )

    parser.add_argument(
        "--layers",
        type=int,
        help="How many layer of the HRL",
        default=1,
    )

    parser.add_argument(
        "--her",
        action='store_true',
        help='if use her or not',
    )

    parser.add_argument(
        '--imitation',
        action='store_true',
        help='if use imitation learning'
    )

    parser.add_argument(
        '--imit_ratio',
        type=float,
        default=1.0,
        help='ratio of imitation loss by actor loss'
    )

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

    parser.add_argument(
        '--rtype',
        type=str,
        default="sparse",
        help='sparse reward or dense reward'
    )

    parser.add_argument(
        '--curriculum',
        type=int,
        default=0,
        help='num of curriculums'
    )

    parser.add_argument(
        '--curiosity',
        action='store_true',
        help='use curiosity driven method'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.threadings > 1:  # otherwise there will be a bug
        FLAGS.show = False

    if FLAGS.test:
        FLAGS.retrain = False

    print("You can check your parameters here. Press 'c' to continue.")
    print(FLAGS)

    import pdb; pdb.set_trace()
    return FLAGS
