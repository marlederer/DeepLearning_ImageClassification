import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go GTN")
    parser.add_argument('--batch', type=int, default=64,
                        help="the batch size training procedure")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--keepprob', type=float, default=0.5)
    parser.add_argument('--keepprobconv', type=float, default=0.7)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--mu', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    parser.add_argument('--dataset', type=str, default='cifar-10',
                        help="available datasets: [cifar-10, gtsrb]")
    parser.add_argument('--model', type=str, default='hist', help='classification-model, support [hist, bovw, LeNet, VGGnet]')
    parser.add_argument('--augmentation', type=int, default=0)
    return parser.parse_args()
