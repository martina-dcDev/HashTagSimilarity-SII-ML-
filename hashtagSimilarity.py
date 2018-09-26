import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from sklearn.decomposition import PCA
from matplotlib import pyplot
import glob, re

#path of the directory where dataset is stored
directory_path = r'C:\Users\skunkMarty\instaloader\profili\*\*.txt'
#regex to retrieve hashtags from captions
hastag_regex = r'(\#[a-zA-Z]+\b)(?!;)'
#list of lists: each list is the set of hashtags related to a photo in the dataset
hashtags_lists = []
#map(k,v) where k are indexes of hashtags_lists and v are the paths of pictures
indexName_map = {}
# ----- START: MODEL PARAMETERS -----
myModelPath = 'Model1'
mySize = 150
myWindow = 10
myMinCount = 5
myWorkers = 10
myEpochs = 10
# ----- END: MODEL PARAMETERS -----
saveHashtags = True


def getHashtags(dataset_path,regex):
    index = 0
    files_paths = glob.glob(dataset_path)
    for file_path in files_paths:
        file = open(file_path,'r',encoding="utf8")
        readFile = file.read()
        file_hashtags = re.findall(regex,readFile)
        file.close()
        if len(file_hashtags) != 0:
            hashtags_lists.append(file_hashtags)
            indexName_map[str(index)] = file_path
            index = index + 1
    return hashtags_lists

def trainModel(list_of_lists,model_path,mySize,myWindow,myMin_count,myWorkers,myEpochs):
    model = gensim.models.Word2Vec(list_of_lists, size= mySize, window=myWindow,min_count=myMin_count,workers=myWorkers)
    model.train(list_of_lists, total_examples=len(list_of_lists),epochs=myEpochs)
    model.save(model_path)
    return model

def loadModel(model_path):
    model = model = gensim.models.Word2Vec.load(model_path)
    return model

def main():
    h_lists = getHashtags(directory_path,hastag_regex)
    model = trainModel(h_lists,myModelPath,mySize,myWindow,myMinCount,myWorkers,myEpochs)
    hashtags = sorted(model.wv.vocab.keys())
    if saveHashtags is True:
      # Save hashtags to file: hashtags.txt
      fp = open('hashtags-dataset1.txt', 'r+', encoding="utf-8")
      for hashtag in hashtags:
       fp.write(hashtag + '\n')
      fp.close()

    result1 = model.similarity('#lago','#salmone')
    result2 = model.similarity('#pizza','#cucina')
    result3 = model.similarity('#pizza','#risotto')
##    result4 = model.similarity('#tattoo','#art')
##    result5 = model.similarity('#pasta','#cucina')
##    result6 = model.similarity('#roma','#milano')
    w2=["#roma"] 
    result_w2 = model.wv.most_similar(positive=w2,topn=5)
    w3=["#green"] 
    result_w3 = model.wv.most_similar(positive=w3,topn=5)
    print("Most similar Hashtags for #roma")
    print(result_w2)
    print("Most similar Hashtags for #green")
    print(result_w3)
    print("similarity ['#lago','#salmone']:")
    print(result1)
    print("similarity ['#pizza','#cucina']:")
    print(result2)
    print("similarity ['#pizza','#risotto']:")
    print(result3)
##    print("similarity ['#tattoo','#art']:")
##    print(result4)
##    print("similarity ['#pasta','#cucina']:")
##    print(result5)
##    print("similarity ['#roma','#milano']:")
##    print(result6)
    

main()
    
