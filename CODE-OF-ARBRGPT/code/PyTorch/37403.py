import torch

from collections import Counter



if __name__ == '__main__':

    for corpus_size in [10000, 1000000]:

        print('when corpus size={}'.format(corpus_size))

        for device in ['cpu', 'cuda']:

            freqs = [1.0 for _ in range(corpus_size)]

            freqs = torch.tensor(freqs, device=device)

            samples = []

            for _ in range(100):

                samples += torch.multinomial(freqs, 100000, replacement=True).tolist()

            counter = Counter(samples)

            counter = {k: v for k, v in

                       sorted(dict(counter).items(), key=lambda item: item[0], reverse=False)}

            keys = list(counter.keys())

            values = list(counter.values())

            print('  in devce {}'.format(device))

            print('\tkeys:', keys[:10])

            print('\tcount:', values[:10])