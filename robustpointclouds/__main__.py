import argh
from robustpointclouds.commands import evaluate

parser = argh.ArghParser()
parser.add_commands([evaluate])

if __name__ == '__main__':
    parser.dispatch()
