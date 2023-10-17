import argh
from robustpointclouds.commands import evaluate_loss, predict, visualize_loss

parser = argh.ArghParser()
parser.add_commands([evaluate_loss, predict, visualize_loss])

if __name__ == '__main__':
    parser.dispatch()
