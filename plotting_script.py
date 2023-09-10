import argparse
import os

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
        mode = log_dir.split("True-")[1]

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


def plot_tb(run_ids, tags, convert_tb_to_csv=False):
    # run_ids = "20000_one-hot/games_0-20000-encoding_one-hot-point_rewards_True-Tue_Sep__5_00-02-44_2023"

    df = pd.DataFrame([])

    mode = ["Masking", "No Masking"]
    i = 0

    for run_id in run_ids:
        log_dir = f'tensorboard_graphs/{run_id}'

        log_df = tensorboard_logs_to_df(log_dir, mode[i])
        i += 1

        df = pd.concat([df, log_df])

    # select all tags
    if tags is None:
        tags = df['tag']

    for tag_name in tags:
        df_tag = df[df['tag'] == tag_name].explode(['step', 'value'])

        output_csv_file = f'plots/{tag_name}_{run_ids}.csv'

        if convert_tb_to_csv:
            df.to_csv(output_csv_file, index=False)

        sns.lineplot(
            data=df_tag,
            x='step',
            y='value',
            hue='Encoding',
        )

        plt.xlabel('Step')
        plt.ylabel(tag_name)
        # plt.title(f'{tag_name} over Steps')
        plt.show()


# one-hot_comp,eval/loss: ,[60000],[0.9782999753952026]
# mixed,eval/loss: ,[60000],[1.0037000179290771]
# mixed_comp,eval/loss: ,[60000],[1.0735000371932983]
# one-hot,eval/loss: ,[60000],[0.960099995136261]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tensorboard data to csv.')

    run_ids = ["games_0-20000-mask_loss_output-240-no_gas-Sun_Sep_10_01-58-14_2023",
               "games_0-20000-not_masked-240-no_gas-Sun_Sep_10_02-07-09_2023",
               # "20000_one-hot/games_0-20000-encoding_one-hot-point_rewards_True-Tue_Sep__5_23-20-39_2023",
               ]

    tags = [
        "train/probability_of_correct_action", "train/loss", "eval/prob_correct_action", "train/rate_oob_actions",
           "train/rate_wrong_action_taken",
        "eval/prob_correct_action", "eval/rate_wrong_action_taken",
        "eval/loss"
    ]

    parser.add_argument(
        "--run_ids", type=str, nargs="+", default=run_ids,
        help="Run ids of the experiments. The run id is the name of the folder containing the tensorboard files.",
    )
    parser.add_argument(
        "--input-folder", "-i", type=str, default=os.path.join(os.path.dirname(SKAT_DT_ROOT), "training-outputs"),
        help="Path to runs folder of the human-robot-gym repository.",
    )
    parser.add_argument(
        "--output-folder", "-o", type=str, default="csv/raw",
        help="Path to output folder. The csv files will be saved in this folder.",
    )
    parser.add_argument(
        "--tags", "-t", type=str, nargs="*", default=tags,
        help="List of tags to include in the csv file. If not specified, all tags will be included.",
    )
    parser.add_argument(
        "--merged", "-m", action="store_true",
        help="If specified, the csv files off all tags will be merged into one file.",
    )

    args = vars(parser.parse_args())

    plot_tb(
        run_ids=args['run_ids'],
        tags=args['tags']
    )
