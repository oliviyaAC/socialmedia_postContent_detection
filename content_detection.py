import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    "caption": [
        "Enjoying my beach vacation! #fun",
        "Check out this violent video",
        "Delicious homemade pizza",
        "I hate everything!",
        "Happy birthday to my friend!",
        "Shocking fight scene #violence",
        "Nature is amazing",
        "This is a terrible day",
        "Loving my new cat #cute",
        "Angry at the traffic jam",
        "Sunset is beautiful",
        "This movie is horrifying",
        "Yummy chocolate cake",
        "Violent fight in the streets",
        "Relaxing day at the park",
        "I am so sad and upset",
        "Cooking my favorite pasta",
        "Watch this scary scene",
        "Happy moments with family",
        "Anger management tips",
        "Awesome workout session",
        "Horrible road accident video",
        "Beautiful flowers in the garden",
        "Terrible behavior in class",
        "Delicious burger recipe",
        "This video contains violence",
        "Relaxing morning yoga",
        "Angry at the news today",
        "Cute puppy playing",
        "Violent fight clip trending"
    ],
    # 1 = Approved, 0 = Not Approved
    "approved": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)


data = {
    "caption": [
        "Enjoying my beach vacation! #fun",
        "Check out this violent video",
        "Delicious homemade pizza",
        "I hate everything!",
        "Happy birthday to my friend!",
        "Shocking fight scene #violence",
        "Nature is amazing",
        "This is a terrible day",
        "Loving my new cat #cute",
        "Angry at the traffic jam",
        "Sunset is beautiful",
        "This movie is horrifying",
        "Yummy chocolate cake",
        "Violent fight in the streets",
        "Relaxing day at the park",
        "I am so sad and upset",
        "Cooking my favorite pasta",
        "Watch this scary scene",
        "Happy moments with family",
        "Anger management tips",
        "Awesome workout session",
        "Horrible road accident video",
        "Beautiful flowers in the garden",
        "Terrible behavior in class",
        "Delicious burger recipe",
        "This video contains violence",
        "Relaxing morning yoga",
        "Angry at the news today",
        "Cute puppy playing",
        "Violent fight clip trending"
    ],
    # 1 = Approved, 0 = Not Approved
    "approved": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)


X_train, X_test, y_train, y_test = train_test_split(
    df['caption'], df['approved'], test_size=0.2, random_state=42
)

# Step 5: Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train classifier
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# Step 7: Evaluate prototype
y_pred = clf.predict(X_test_tfidf)
print("Prototype Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Test AI on new Instagram posts
new_posts = [
    "Amazing sunset at the beach!",
    "Watch this violent fight scene",
    "Delicious chocolate cake recipe",
    "I am so angry today",
    "Relaxing yoga morning",
    "Terrible accident video"
]

new_tfidf = vectorizer.transform(new_posts)
predictions = clf.predict(new_tfidf)

print("\n✅ New Post Predictions:")
for post, pred in zip(new_posts, predictions):
    print(f"Post: '{post}' --> Approved: {bool(pred)}")

# Step 9: Visualization - Approved vs Not Approved
df['approved_label'] = df['approved'].map({1:'Approved', 0:'Not Approved'})
df['approved_label'].value_counts().plot(kind='bar', color=['green','red'])
plt.title("Approved vs Not Approved Posts")
plt.ylabel("Number of Posts")
plt.show()
