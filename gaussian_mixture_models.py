import pandas as pd

colnames = ['area', 'perimeter', 'compactness', 'lenghtOfKernel', 'widthOfKernel', 'assymetryCoefficient',
        'lenghtOfKernelGrove', 'classifier']

data = pd.read_table("seeds_dataset.txt", index_col=False, names=colnames, header=None, delimiter='\t\t|\t')

numpy_array = data.values
label = numpy_array[:, 7]   # The last culomn. The class label.
features = numpy_array[:, :7]   # From index 0 to 6. 7 feature types.

print(features)
print(label)
