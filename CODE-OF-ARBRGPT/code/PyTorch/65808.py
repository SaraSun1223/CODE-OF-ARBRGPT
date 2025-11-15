from torch.utils.data.datapipes.iter import IterableWrapper, TarArchiveReader, FileLoader, Shuffler

# Example tar archive: https://github.com/pytorch/data/blob/main/examples/vision/fakedata/caltech101/101_ObjectCategories.tar.gz
dp = IterableWrapper(["101_ObjectCategories.tar.gz"])
dp = FileLoader(dp)
dp = TarArchiveReader(dp)
dp = Shuffler(dp, buffer_size=1000)  # Buffer size larger than the number of files in the tar archive

_, buffer = next(iter(dp))
buffer.read()