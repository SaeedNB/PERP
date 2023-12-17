from numba import jit

@jit
def find_first_occurrence(start, node, charge_time, size, arr):
    for i in range(start, size, 5):
        result = True
        for j in range(charge_time):
            result = result and (arr[i + j] == 0)
            if not result:
                break

        if result:
            return i - start
    return 0