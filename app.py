from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("poly.pkl", "rb") as file:
    poly = pickle.load(file)

print("✅ Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        likes = float(request.form["likes"])
        comments = float(request.form["comments"])
        shares = float(request.form["shares"])
        saves = float(request.form["saves"])
        profile_visits = float(request.form["profile_visits"])
        follows = float(request.form["follows"])

        # Prevent negative values
        if min(
            likes,
            comments,
            shares,
            saves,
            profile_visits,
            follows
        ) < 0:

            return render_template(
                "index.html",
                error="Values cannot be negative."
            )

        # Same order as ML model
        new_data = [[
            likes,
            comments,
            shares,
            saves,
            profile_visits,
            follows
        ]]

        # Polynomial transformation
        new_data_poly = poly.transform(new_data)

        # Prediction
        prediction = model.predict(new_data_poly)

        impressions = round(float(prediction[0]), 2)

        # Engagement
        total_engagement = (
            likes +
            comments +
            shares +
            saves
        )

        # Engagement rate
        if impressions > 0:
            engagement_rate = (
                total_engagement / impressions
            ) * 100
        else:
            engagement_rate = 0

        engagement_rate = round(
            engagement_rate,
            2
        )

        return render_template(
            "index.html",
            prediction=impressions,
            likes=likes,
            comments=comments,
            shares=shares,
            saves=saves,
            profile_visits=profile_visits,
            follows=follows,
            total_engagement=total_engagement,
            engagement_rate=engagement_rate
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


@app.route("/chat", methods=["POST"])
def chat():

    question = request.form.get("question", "").lower()

    if "impression" in question:
        answer = "Impressions are predicted using Likes, Comments, Shares, Saves, Profile Visits and Follows."

    elif "like" in question:
        answer = "Likes are an important engagement metric for your Instagram post."

    elif "comment" in question:
        answer = "Comments show how users interact with your Instagram content."

    elif "share" in question:
        answer = "Shares show that people found your content useful enough to share."

    elif "save" in question:
        answer = "Saves show that users want to keep your content for later."

    elif "increase" in question or "improve" in question:
        answer = "To improve impressions, focus on increasing Likes, Shares, Saves and Profile Visits."

    elif "model" in question or "machine learning" in question:
        answer = "This project uses a Machine Learning regression model to predict Instagram Impressions."

    else:
        answer = "I can help you understand Instagram Impressions, Likes, Comments, Shares, Saves and Engagement."

    return render_template(
        "index.html",
        chat_answer=answer
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )


    