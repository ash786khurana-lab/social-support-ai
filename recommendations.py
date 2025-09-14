def generate_recommendations(features, decision):
    recs = []

    # Normalize decision check
    decision_text = str(decision).lower()

    if "approved" in decision_text:
        recs.append(" Provide monthly financial aid.")
        recs.append(" Offer upskilling / training opportunities.")
        recs.append(" Support housing / rent assistance if required.")
    elif "declined" in decision_text or "soft" in decision_text:
        recs.append(" Application declined, but alternative support is available.")
        recs.append(" Suggest free training & certification programs.")
        recs.append(" Offer job-matching and career counseling.")
    else:
        recs.append(" Unable to determine recommendations due to invalid data.")

    return recs
