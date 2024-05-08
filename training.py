from os import listdir
from os.path import isfile, join

mypath = 'training_material'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# Filters the files which are in the html format
onlyfiles = [x for x in onlyfiles if 'htm' in x]

# there are plenty of HTMLLoaders to choose from
from langchain_community.document_loaders import UnstructuredHTMLLoader

# the data dictionary will contain the documents
data = {}

i = 0
for file in onlyfiles:
    loader = UnstructuredHTMLLoader(file)
    data[i] = loader.load()
    i += 1

