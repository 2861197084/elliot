

from scipy import stats
import typing as t


class PairedTTest:
    @staticmethod
    def common_users(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float]):
        return list(arr_0.keys() & arr_1.keys())

    @staticmethod
    def compare(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float], users: t.List[int]):
        list_0 = list(map(arr_0.get, users))
        list_1 = list(map(arr_1.get, users))
        return stats.ttest_rel(list_0, list_1)[1]

