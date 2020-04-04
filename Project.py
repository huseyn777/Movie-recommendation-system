import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from scipy import spatial



def readFromFile(movie_id, movie_rating, input_from_file):
    for x in range(1, len(input_from_file)):
        line = input_from_file[x]
        tempStr = ""
        firstNonDigit = 0

        for y in range(0, len(line)):
            if line[y].isdigit():
                tempStr = tempStr + line[y]
            else:
                firstNonDigit = y
                break
        movie_id.append(tempStr)
        tempStr = ""

        for y in range(firstNonDigit, len(line)):
            if line[y].isdigit():
                firstNonDigit = y
                break

        for y in range(firstNonDigit, len(line)):
            if line[y].isdigit():
                tempStr = tempStr + line[y]
            else:
                firstNonDigit = y
                break
        movie_rating.append(tempStr)

    for i in range(0, len(movie_id)):
        movie_id[i] = int(movie_id[i])
        movie_rating[i] = int(movie_rating[i])


def createGenreVector(genresName, genresVector, genres, movie_id, movie_rating, weigthAvr):
    genresCount = []
    for i in range(len(movie_id)):
        temp = genres[i]
        temp = temp.split(",")
        for j in range(len(temp)):
            if temp[j] not in genresName:
                genresName.append(temp[j])
                genresCount.append(1)
                genresVector.append(movie_rating[i] - weigthAvr)
            else:
                index = genresName.index(temp[j])
                genresCount[index] = genresCount[index] + 1
                genresVector[index] = genresVector[index] + \
                    (movie_rating[i] - weigthAvr)

    for i in range(len(genresVector)):
        genresVector[i] = genresVector[i] / genresCount[i]


def createNconstector(nconstName, nconstVector, nconst, movie_id, movie_rating, weigthAvr):
    nconstCount = []
    for i in range(len(movie_id)):
        temp = nconst[i]
        temp = temp.split(",")
        for j in range(len(temp)):
            if temp[j] not in nconstName:
                nconstName.append(temp[j])
                nconstCount.append(1)
                nconstVector.append(movie_rating[i] - weigthAvr)
            else:
                index = nconstName.index(temp[j])
                nconstCount[index] = nconstCount[index] + 1
                nconstVector[index] = nconstVector[index] + \
                    (movie_rating[i] - weigthAvr)

    for i in range(len(nconstVector)):
        nconstVector[i] = nconstVector[i] / nconstCount[i]


def createRegionVector(regionName, regionVector, region, movie_id, movie_rating, weigthAvr):
    regionCount = []
    for i in range(len(movie_id)):
        temp = region[i]
        temp = temp.split(",")
        for j in range(len(temp)):
            if temp[j] not in regionName:
                regionName.append(temp[j])
                regionCount.append(1)
                regionVector.append(movie_rating[i] - weigthAvr)
            else:
                index = regionName.index(temp[j])
                regionCount[index] = regionCount[index] + 1
                regionVector[index] = regionVector[index] + \
                    (movie_rating[i] - weigthAvr)

    for i in range(len(regionVector)):
        regionVector[i] = regionVector[i] / regionCount[i]
    for i in range(len(regionName)):
        if(regionName[i] == "\\N"):
            regionVector.pop(i)
    regionName.remove("\\N")


def collaborative(movie_id):
    m1 = [2,"x",3]
    m2 = ["x",9,3]
    m3 = [5,7,"x"]
    m4 = ["x",7,8]
    m5 = [1,"x",3]
    m6 = [4,10,"x"]

    collaborativeTable = [[0 for x in range(len(m1))] for y in range(len(movie_id))] 

    collaborativeTable[0] = m1
    collaborativeTable[1] = m2
    collaborativeTable[2] = m3
    collaborativeTable[3] = m4
    collaborativeTable[4] = m5
    collaborativeTable[5] = m6


    weight = 0
    count = 0

    vector = [[0 for x in range(len(m1))] for y in range(len(movie_id))] 

    for i in range(0,len(collaborativeTable)):
        for j in range(len(collaborativeTable[i])):
            vector[i][j] = collaborativeTable[i][j]

    for i in range(0,len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j] != "x":
                weight = weight + vector[i][j]
                count = count + 1
        weight = weight/count

        for j in range(len(vector[i])):
            if vector[i][j] != "x":
                vector[i][j] = vector[i][j] - weight
        weight = 0
        count = 0     

    for i in range(0,len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j] == "x":
                vector[i][j] = 0

    cosSim = [[0 for x in range(len(movie_id))] for y in range(len(movie_id))] 


    for i in range(0,len(cosSim)):
        for j in range(len(cosSim[i])):
            cosSim[i][j] = 1 - spatial.distance.cosine(vector[i], vector[j])

    print(collaborativeTable)

    for i in range(0,len(collaborativeTable)):
        for j in range(len(collaborativeTable[i])):
            if collaborativeTable[i][j] == "x":
                vector[i][j] = collaborativeTable[i][j]

    rating = 0
    count = 0

    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j] == "x":
                for z in range(len(cosSim[i])):
                    if z != i:
                        if cosSim[i][z] > 0:
                            if(vector[z][j] != "x"):
                                count = count + cosSim[i][z]
                                rating = rating + (cosSim[i][z] * collaborativeTable[z][j])   
                if(count != 0):
                    collaborativeTable [i][j] = rating/count
                else:
                    collaborativeTable [i][j] = 0
                count = 0
                rating = 0


    print(collaborativeTable)

all_data = pd.read_csv('all_data.csv', sep='\t', header=0)
data = all_data
del data['Unnamed: 0']
data = data.replace(np.nan, '', regex=True)

file1 = open("Input.txt", "r")
input_from_file = file1.readlines()
file1.close()

movie_id = []
movie_rating = []

# Function to read input from file
readFromFile(movie_id, movie_rating, input_from_file)
weigthAvr = sum(movie_rating) / len(movie_rating)


user_movies = data.loc[[movie_id[0]]]
for i in range(1, len(movie_id)):
    temp = data.loc[[movie_id[i]]]
    user_movies = user_movies.append(temp, ignore_index=True)

genres = user_movies['genres']
genresName = []
genresVector = []
createGenreVector(genresName, genresVector, genres, movie_id, movie_rating, weigthAvr)

nconst = user_movies['nconst']
nconstName = []
nconstVector = []
createNconstector(nconstName, nconstVector, nconst, movie_id, movie_rating, weigthAvr)

region = user_movies['region']
regionName = []
regionVector = []
createRegionVector(regionName, regionVector, region, movie_id, movie_rating, weigthAvr)



#########################################################################
######################## Global Similarity ##############################
# creating vector for cosine similarity
data_lng5 = pd.DataFrame(data[['index']])
df_sim5 = pd.DataFrame(0, index=range(
    0, data_lng5.size), columns=['mean'])



#########################################################################
##################### Similarity of genres ###########################
df = pd.DataFrame([genresName])
df_values = pd.DataFrame([genresVector])
data_lng = pd.DataFrame(data[['genres']])

# df_new is matrix that is used to generate similarities
df_new = pd.DataFrame('', index=range(
    0, data_lng.size), columns=range(0, 0))

# creating vector for cosine similarity
df_sim = pd.DataFrame(0, index=range(
    0, data_lng.size), columns=['Sim'])

# filling values to values matrix(1,0) to compute cosine similarity
for index, rows in df.iteritems():
    df_new[rows[0]] = data_lng['genres'].str.contains(rows[0], na=False)

df_new = df_new.astype(int)

# calculating cosine similarity
df_sim = pd.DataFrame(cosine_similarity(df_new, df_values))
df_sim.columns = ['Sim']



#########################################################################
####################### Similarity of crew ##############################
df2 = pd.DataFrame([nconstName])
df_values2 = pd.DataFrame([nconstVector])
data_lng2 = pd.DataFrame(data[['nconst']])

# df_new is matrix that is used to generate similarities
df_new2 = pd.DataFrame('', index=range(
    0, data_lng2.size), columns=range(0, 0))

# creating vector for cosine similarity
df_sim2 = pd.DataFrame(0, index=range(
    0, data_lng2.size), columns=['Sim'])

# filling values to values matrix(1,0) to compute cosine similarity
for index, rows in df2.iteritems():
    df_new2[rows[0]] = data_lng2['nconst'].str.contains(rows[0], na=False)

df_new2 = df_new2.astype(int)

# calculating cosine similarity
df_sim2 = pd.DataFrame(cosine_similarity(df_new2, df_values2))
df_sim2.columns = ['Sim2']



#########################################################################
##################### Similarity of languages ###########################
df3 = pd.DataFrame([regionName])
df_values3 = pd.DataFrame([regionVector])
data_lng3 = pd.DataFrame(data[['region']])

# df_new is matrix that is used to generate similarities
df_new3 = pd.DataFrame('', index=range(
    0, data_lng3.size), columns=range(0, 0))

# creating vector for cosine similarity
df_sim3 = pd.DataFrame(0, index=range(
    0, data_lng3.size), columns=['Sim'])

# filling values to values matrix(1,0) to compute cosine similarity
for index, rows in df3.iteritems():
    df_new3[rows[0]] = data_lng3['region'].str.contains(rows[0], na=False)

df_new3 = df_new3.astype(int)

# calculating cosine similarity
df_sim3 = pd.DataFrame(cosine_similarity(df_new3, df_values3))
df_sim3.columns = ['Sim3']



#########################################################################
##################### Summing Up ###########################
df_sim = pd.concat([df_sim, df_sim2], axis=1)
df_sim = pd.concat([df_sim, df_sim3], axis=1)
df_sim = pd.concat([df_sim, df_sim5], axis=1)


for i in range(1, len(movie_id)):
    temp = movie_id[i]
    df_sim = df_sim.drop(temp)


df_sim['mean'] = df_sim.mean(axis=1)
df_sim = df_sim.sort_values(by=['mean'], ascending=False)
df_sim = df_sim.drop('Sim', axis=1)
df_sim = df_sim.drop('Sim2', axis=1)
df_sim = df_sim.drop('Sim3', axis=1)
df_sim = df_sim.head(20)

df_sim['name'] = ''

for index, rows in df_sim.iterrows():
    df_sim['name'][index] = all_data['primaryTitle'][index]

#Collaborative filtering(item-item) part -> optional
print("Collaborative filtering(item-item):")
#collaborative(movie_id)

print("")

print("Content based:")
print(df_sim)

