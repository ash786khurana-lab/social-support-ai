from ingestion import ingest_all
from eligibility import build_features, check_eligibility
from recommendations import generate_recommendations   # 👈 new import

data = ingest_all()
features = build_features(data)
decision = check_eligibility(features)
recs = generate_recommendations(features, decision)     # 👈 new call

print("Extracted Features:", features)
print("Final Decision:", decision)
print("\nRecommendations:")
for r in recs:
    print("-", r)
