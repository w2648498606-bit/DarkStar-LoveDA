import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import train as base
import dg_sweep_common as dg
import e037_next_common as nx

RUN_NAME = 'E090_e037_blv_lite'
METHOD = 'E037 + BLV-lite class-frequency-dependent training-time logit variation'


def parse_args():
    parser = argparse.ArgumentParser(description='Run the final E090 training recipe.')
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help='Path to the LoveDA dataset root containing Train/ and Val/.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    run_epoch = nx.make_blv_run_epoch(tau=0.50, sample_masks=128)
    original_dataset = dg.setup_run(RUN_NAME, balanced=False, train_run_epoch=run_epoch)
    sys.argv = nx.e037_argv(RUN_NAME, epochs=12)
    sys.argv = nx.replace_argv_value(sys.argv, '--data-root', str(data_root))
    print('E090 | E037 + BLV-lite | tau=.50 | class frequencies estimated from 128 Train masks')
    base.main()
    dg.finish_run(RUN_NAME, METHOD, original_dataset)


if __name__ == '__main__':
    main()
