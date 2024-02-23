import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go GTN")
    parser.add_argument('--bpr_batch', type=int, default=64,
                        help="the batch size training procedure")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing, 100")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="gtn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    parser.add_argument('--prop_dropout', type=float, default=0.1)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--ogb', type=bool, default=True)
    parser.add_argument('--incnorm_para', type=bool, default=True)

    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--alpha1', type=float, default=0.25)
    parser.add_argument('--alpha2', type=float, default=0.25)

    parser.add_argument('--lambda2', type=float, default=4.0) #2, 3, 4,...

    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    parser.add_argument('--dataset', type=str, default='cifar-10',
                        help="available datasets: [cifar-10, gtsrb]")
    parser.add_argument('--model', type=str, default='hist', help='classification-model, support [hist, bovw, LeNet, VGGnet]')
    parser.add_argument('--avg', type=int, default=0)
    parser.add_argument('--recdim', type=int, default=256,
                        help="the embedding size of GTN: 128, 256")
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--gcn_model', type=str,
                        default='GTN', help='GTN')
    return parser.parse_args()