from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset into a DataFrame
df = pd.read_csv('C:/Users/avnis/PycharmProjects/flaskProject/indian_food.csv')


def recommend_dishes_by_flavor_time(flavor_profile, time_limit):
    filtered_df = df[(df['flavor_profile'] == flavor_profile) & (df['prep_time'] + df['cook_time'] <= time_limit)]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_df['ingredients'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = filtered_df.index
    indices_map = pd.Series(indices, index=filtered_df['name']).drop_duplicates()
    similarity_scores = list(enumerate(cosine_similarities))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1].max(), reverse=True)
    recommended_dishes = [filtered_df['name'].iloc[idx] for idx, _ in sorted_scores[1:6]]
    return recommended_dishes


def recommend_similar_dishes(dish_name):
    dish = df[df['name'] == dish_name]
    if dish.empty:
        print(f"Sorry, '{dish_name}' not found in the dataset.")
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['ingredients'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    index = dish.index[0]
    similarity_scores = list(enumerate(cosine_similarities[index]))

    # Sort the similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 recommended dish names (excluding the input dish)
    recommended_dishes = [df['name'].iloc[idx] for idx, _ in sorted_scores[1:6]]

    return recommended_dishes


app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    flavor_profile = request.form['flavor_profile']
    time_limit = int(request.form['time_limit'])
    recommendations = recommend_dishes_by_flavor_time(flavor_profile, time_limit)
    session['recommendations'] = recommendations  # Store recommendations in session
    return render_template('recommend.html', recommendations=recommendations)


@app.route('/similar', methods=['POST'])
def similar():
    dish_name = request.form['dish_name']
    recommendations = recommend_similar_dishes(dish_name)
    return render_template('similar.html', recommendations=recommendations)


@app.route('/ingredients', methods=['POST'])
def ingredients():
    selection = int(request.form['selection'])
    recommendations = session.get('recommendations')  # Retrieve recommendations from session
    if recommendations and selection in range(1, len(recommendations) + 1):
        selected_dish = recommendations[selection - 1]
        selected_row = df[df['name'] == selected_dish]
        ingredients = selected_row['ingredients'].iloc[0]
        return render_template('ingredients.html', dish=selected_dish, ingredients=ingredients)
    else:
        return render_template('invalid.html')


if __name__ == '__main__':
    app.run(debug=True)
