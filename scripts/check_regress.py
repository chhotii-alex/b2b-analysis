from argparse import ArgumentParser
import subprocess
import pickle
from analysis import get_metacommunity

parser = ArgumentParser(
    prog="check_regress",
    description="does regression test on analysis")
parser.add_argument('-m', '--mode', choices=['pre', 'post'],
                    help='Run with --mode pre to define baseline; run with --mode post to test that results are same as baseline.')
args = parser.parse_args()

_, effective_counts = get_metacommunity(8)

if args.mode == 'pre':
    with open('prev.pkl', 'wb') as file:
        pickle.dump(effective_counts, file)
else:
    with open('prev.pkl', 'rb') as file:
        prev_counts = pickle.load(file)
    assert effective_counts.shape == prev_counts.shape
    assert (effective_counts != prev_counts).max() == False
print("Done! Good!")
subprocess.run(['say', 'Done!'])
