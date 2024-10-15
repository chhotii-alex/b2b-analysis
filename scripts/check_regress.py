import platform
from argparse import ArgumentParser
import subprocess
import pickle
import numpy as np
from analysis import get_metacommunity

parser = ArgumentParser(
    prog="check_regress",
    description="does regression test on analysis")
parser.add_argument('-m', '--mode', choices=['pre', 'post'],
                    help='Run with --mode pre to define baseline; run with --mode post to test that results are same as baseline.')
parser.add_argument('-n', '--number', type=int,
                    default=8,
                    help='number of communities')
args = parser.parse_args()

filename = 'prev%d.pkl' % args.number

_, effective_counts, abundances = get_metacommunity(args.number)

if args.mode == 'pre':
    with open(filename, 'wb') as file:
        pickle.dump(effective_counts, file)
else:
    with open(filename, 'rb') as file:
        prev_counts = pickle.load(file)
    assert effective_counts.shape == prev_counts.shape
    assert np.isclose(effective_counts.toarray(), prev_counts.toarray()).all()
print("Done! Good!")
if platform.system() == 'Darwin':  # This only set up on Mac
    subprocess.run(['say', 'Done!'])
