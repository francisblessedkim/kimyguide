"""Command-line entrypoint for the KimyGuide TF-IDF prototype.

This small script demonstrates a day-one interaction: the user types a
free-text learning goal and the script prints the top-N course
recommendations produced by the TF-IDF recommender, together with a
short explanation for each recommendation.

Usage: run this file as a script (python run_cli.py) and follow the
prompt. It relies on the package `src.kimyguide` to load data and run
the model.
"""

from src.kimyguide.data.loader import load_courses
from src.kimyguide.models.tfidf_recommender import TfidfGoalRecommender
from src.kimyguide.explain.simple_explainer import add_explanations


def main():
    """Run the simple interactive CLI.

    Steps:
      1. Prompt the user for a free-text learning goal.
      2. Load course items from CSV.
      3. Build / use a TF-IDF recommender to score items.
      4. Attach simple explanations and print top-k results.
    """
    print("=== KimyGuide: Day-One Goal-based Recommender (TF-IDF Prototype) ===\n")

    # Read one-line free-text goal from the user
    goal = input("Describe your learning goal: ").strip()
    if not goal:
        # Guard: nothing to recommend if the user provides no goal
        print("No goal entered. Exiting.")
        return

    # Load items (by default this looks for data/raw/courses_mooclike.csv)
    courses = load_courses()

    # Create and use the TF-IDF recommender. This class fits a
    # TF-IDF vectorizer on the course texts and computes cosine
    # similarity between the user's goal and each item.
    model = TfidfGoalRecommender(courses)
    recs = model.recommend(goal_text=goal, top_k=5)

    # Add human-readable explanations to the returned candidates
    recs = add_explanations(recs)

    # Present results
    print("\nTop 5 recommended items:\n")
    for _, row in recs.iterrows():
        title = row.get("title", "[no title]")
        print(f"- {title} (score={row['score']:.4f})")
        print(f"  {row['explanation']}\n")


if __name__ == "__main__":
    main()
