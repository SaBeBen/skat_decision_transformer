import argparse
import os
import time

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SKAT_DT_ROOT = os.getcwd()


def tensorboard_logs_to_df(log_dir, mode=None):
    event_acc = event_accumulator.EventAccumulator(log_dir).Reload()

    data = []

    tags = event_acc.Tags()['scalars']

    # exclude not used metrics
    tags = tags[:13]

    if mode is None:
        mode = log_dir.split("encoding_")[1].split("-po")[0]
        #.split("-wo_mask")[0].capitalize().replace("_", " ").replace("comp", "comp.")

    for tag in tags:
        events = event_acc.Scalars(tag)
        data.append({
            'Encoding': mode,
            'tag': tag,
            'step': [int(event.step) for event in events],
            'value': [float(event.value) for event in events]
        })

    df = pd.DataFrame(data)
    return df


def plot_tb(run_ids, tags, name, output_dir=None, convert_tb_to_csv=False):
    df = pd.DataFrame([])

    i = 0

    for run_id in run_ids:
        log_dir = f'tensorboard_graphs/{run_id}'

        log_df = tensorboard_logs_to_df(log_dir, None)
        i += 1

        df = pd.concat([df, log_df])

    # select all tags
    if tags is None:
        tags = df['tag']

    for tag_name in tags:
        df_tag = df[df['tag'] == tag_name].explode(['step', 'value'])

        output_csv_file = f'{output_dir}/csv/{name}_{tag_name}.csv'

        if convert_tb_to_csv and not (output_dir is None):
            if not os.path.exists(f"{output_dir}/csv/{name}"):
                os.makedirs(f"{output_dir}/csv/{name}")
            df.to_csv(output_csv_file, index=False)

        sns.lineplot(
            data=df_tag,
            x='step',
            y='value',
            # hue='Encoding',
        )

        plt.xlabel('Step')
        ylabel = tag_name.split("/")[1].replace("_", " ").replace(": ", "").replace("prob ", "probability ").capitalize()
        plt.ylabel(ylabel)

        new_xticks = np.arange(0, 1200001, 400000)  # Adjust the step size here
        plt.xticks(new_xticks)

        if not (output_dir is None):
            # Create dir if not existent
            if not os.path.exists(f'{output_dir}/plots/{name}_{tag_name.split("/")[0]}'):
                os.makedirs(f'{output_dir}/plots/{name}_{tag_name.split("/")[0]}')

            plt.savefig(f'{output_dir}/plots/{name}_{tag_name}_{name}_plot.svg',
                        format='svg',
                        bbox_inches='tight')
        # plt.title(f'{tag_name} over Steps')
        plt.show()


# one-hot_comp,eval/loss: ,[60000],[0.9782999753952026]
# mixed,eval/loss: ,[60000],[1.0037000179290771]
# mixed_comp,eval/loss: ,[60000],[1.0735000371932983]
# one-hot,eval/loss: ,[60000],[0.960099995136261]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Tensorboard logs')

    # possible existing run ids
    run_ids = [
        "games_0--1-encoding_one-hot-point_rewards_True-card_put-pure_loss-Thu_Sep__7_22-41-35_2023"
        # "games_0-10000-one-hot-wo_mask-card_put-Mon_Sep_11_23-59-52_2023",
        # "games_0-10000-mixed-wo_mask-card_put-Mon_Sep_11_23-57-37_2023",
        # "games_0-10000-one-hot_comp-wo_mask-Wed_Sep_13_00-21-03_2023",
        # "games_0-10000-mixed_comp-wo_mask-Wed_Sep_13_00-20-16_2023"
        # "games_0-20000-masked-240-no_gas-Sun_Sep_10_01-58-14_2023",
        # "games_0-20000-not_masked-240-no_gas-Sun_Sep_10_02-07-09_2023",
        # "20000_one-hot/games_0-20000-encoding_one-hot-point_rewards_True-Tue_Sep__5_23-20-39_2023",
    ]

    tags = [
        "train/probability_of_correct_action", "train/loss", "eval/prob_correct_action", "train/rate_oob_actions",
        "train/rate_wrong_action_taken",
        "eval/prob_correct_action", "eval/rate_wrong_action_taken",
        "eval/loss: "
    ]
    parser.add_argument(
        "--tags", nargs="+", type=str, default=tags,
        help="Tags to include in the plots. Should be present in logs.",
    )
    parser.add_argument(
        "--run_ids", type=str, nargs="+", default=run_ids,
        help="Run ids of the experiments. The run id is the name of the folder containing the tensorboard files.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=os.path.join(os.path.dirname(SKAT_DT_ROOT), "training-outputs"),
        help="Directory where runs are stored in tensorboard logging folders.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./plots",
        help="Directory to write to. If none, does not save csv or plot.",
    )
    current_time = time.asctime().replace(':', '-').replace(' ', '_')
    parser.add_argument(
        "--plot_name", type=str, default="all_wc_games",  # f"{current_time}",
        help="Name of plot.",
    )

    args = vars(parser.parse_args())

    plot_tb(
        name=args['plot_name'],
        run_ids=args['run_ids'],
        tags=args['tags'],
        output_dir=args['output_dir'],
    )
