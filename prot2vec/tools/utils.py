from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy.stats import norm

class TrainingProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            progress_type = task.fields.get("progress_type")
            if progress_type == "epochs":
                self.columns = (
                    "[progress.description]{task.description}",
                    BarColumn(),
                     "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    TextColumn("[green]{task.completed} of {task.total} completed", justify="right"),
                    "•",
                    TimeElapsedColumn(),
                )
            elif progress_type == "training":
                self.columns = (
                    "[progress.description]{task.description}",
                    BarColumn(),
                     "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    TextColumn("[green]{task.completed} / {task.total} ", justify="right"),
                    TimeElapsedColumn() if task.finished else TimeRemainingColumn(),
                )
            elif progress_type == "validation":
                self.columns = (
                    "[progress.description]{task.description}",
                    BarColumn(),
                     "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    TextColumn("[green]{task.completed} / {task.total} ", justify="right"),
                    TimeElapsedColumn() if task.finished else TimeRemainingColumn(),
                )
            yield self.make_tasks_table([task])


def save_list_to_file(item_list, filename):
    with open(filename, 'w') as out:
        for item in item_list:
            out.write(str(item) + '\n')


def zscore_to_pvalue(z, two_tailed=False):
    if z > 0:
        p = 1 - norm.cdf(z) 
    else:
        p = norm.cdf(z)
    if two_tailed:
        return 2 * p
    return p


def assert_lexicographical_order(df, p1='protein1', p2='protein2'):
    """
    Guarantees that lexicographical order is maintained in the dataframe so that
    # that df[p1] < df_col[p2]
    :param df: The dataframe to modify
    :param p1: the name of the min column
    :param p2: the name of the max column
    :return: None
    """
    # 3.- we guarantee the lexicographical order between
    # the protein columns, that is,
    # that df_col.protein1 < df_col.protein2
    min_protein = df[[p1, p2]].min(axis=1)
    max_protein = df[[p1, p2]].max(axis=1)
    df.loc[:, p1] = min_protein
    df.loc[:, p2] = max_protein
