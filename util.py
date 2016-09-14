from collections import defaultdict

def is_int(x):
    try:
        int(x)
        return True
    except:
        return False


def FreqDist(tags):
    freq = defaultdict(lambda: 0)
    for tag in tags:
        freq[tag] += 1
    return freq