import time
import random

def select(filename, k=10):
    samples = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()

        random_set = sorted(random.sample(range(file_size), k))

        for i in range(k):
            f.seek(random_set[i])
            f.readline()
            line = f.readline().strip()
            line = line.decode("utf-8").strip()
            try:
                number, rest = line.split(" ", 1)
            except:
                samples.append("This is a test.")

            parts = rest.split('\t')
            samples.append(parts[0])

    assert len(samples)==k, "sample size is not correct"

    return samples

if __name__ == "__main__":
    filename = "/home/rohola/test.txt"
    a = select(filename)
    print(a)