
# Import libraries
import pandas as pd
from scipy.spatial.distance import cosine
from pysptools import distance
import matplotlib.pyplot as plt

# Read the csv files containing the data
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
       'Dec2023_lognormal_priors/'
obs_df = pd.read_csv(path + 'above_water_reflectance_2022_Mobley.csv')  # observation data
sim_df = pd.read_csv(path + "simulated_reflectance.csv")  # simulated data

sam_distance_test = distance.SAM(sim_df["10"], sim_df["10"])
print(sam_distance_test)


# Calculate similarity between two spectra using cosine similarity.
def calculate_cosine_similarity(spectrum1, spectrum2):
    return 1 - cosine(spectrum1, spectrum2)


# Calculate similarity between two spectra using Spectral Angle Mapping.
def calculate_sam_similarity(spectrum1, spectrum2):
    sam_distance = distance.SAM(spectrum1, spectrum2)
    return sam_distance


# Find the most similar spectra
def find_most_similar_spectra(target, dataframe, num_similar=10):
    similarities = dataframe.apply(lambda col1: calculate_sam_similarity(target, col1), axis=0)
    most_similar_columns = similarities.nsmallest(num_similar).index
    most_similar_spectra = dataframe[most_similar_columns]
    similarity_scores = similarities[most_similar_columns]

    return most_similar_columns, most_similar_spectra, similarity_scores


# Find the 10 most similar spectra for each target spectrum in df2
num_similar_spectra = 10
similar_spectra_dict = {}

for target_spectrum_column in obs_df.columns:
    target_spectrum = obs_df[target_spectrum_column]
    most_similar_columns, most_similar_spectra, similarity_scores = find_most_similar_spectra(
        target_spectrum, sim_df, num_similar=num_similar_spectra)
    similar_spectra_dict[target_spectrum_column] = (most_similar_columns, similarity_scores)

# Display the results
for target_spectrum_column, (similar_columns, similarity_scores) in similar_spectra_dict.items():
    print(f"\nTop {num_similar_spectra} most similar spectra for '{target_spectrum_column}':")
    for col, score in zip(similar_columns, similarity_scores):
        print(f" - Spectrum '{col}': Similarity Score = {score:.4f}")

# Save the results to a CSV file
result_df = pd.DataFrame.from_dict(similar_spectra_dict, orient='index',
                                   columns=['MostSimilarSpectra', 'SimilarityScores'])
result_df.to_csv(path + '/checks/sam_similarity_sim_obs.csv')


# Calculate similarity scores for a target spectrum against all spectra in the dataframe
def calculate_all_similarity_scores(target_spectrum, dataframe, similarity_function):
    similarities = dataframe.apply(lambda col: similarity_function(target_spectrum, col), axis=0)
    return similarities


# Calculate similarity scores for the target spectrum against all spectra in the simulated dataset
target_spectrum = sim_df['2843']
all_similarity_scores = calculate_all_similarity_scores(target_spectrum, sim_df,
                                                        similarity_function=calculate_sam_similarity)

# Plot a histogram of the similarity scores
plt.hist(all_similarity_scores, bins=3000, edgecolor='black')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Histogram of Similarity Scores')
plt.show()
