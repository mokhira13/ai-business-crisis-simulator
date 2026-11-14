import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Business Crisis Simulator", page_icon="☕", layout="wide")

TRAINING = [
    ("negotiate supplier discount", "cost_control"),
    ("find cheaper suppliers", "cost_control"),
    ("reduce unnecessary expenses", "cost_control"),
    ("renegotiate rent", "cost_control"),
    ("reduce packaging costs", "cost_control"),
    ("optimize inventory", "cost_control"),
    ("raise coffee prices", "pricing"),
    ("increase menu prices", "pricing"),
    ("offer premium products", "pricing"),
    ("introduce bundle pricing", "pricing"),
    ("change product prices", "pricing"),
    ("offer discounts", "pricing"),
    ("launch social media advertising", "marketing"),
    ("create a loyalty program", "marketing"),
    ("run a promotional campaign", "marketing"),
    ("partner with local influencers", "marketing"),
    ("attract customers with events", "marketing"),
    ("advertise on Instagram", "marketing"),
    ("train employees", "service"),
    ("improve customer service", "service"),
    ("hire additional staff", "service"),
    ("reduce waiting times", "service"),
    ("improve product quality", "service"),
    ("respond to customer complaints", "service"),
]

texts, labels = zip(*TRAINING)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, labels)

CRISES = [
    {
        "title": "Supplier Price Shock",
        "story": "Your coffee supplier increases prices by 30%. Your costs are rising.",
        "choices": [
            ("Negotiate with the supplier", "negotiate supplier discount", -500, 2, 0),
            ("Increase coffee prices", "raise coffee prices", 1800, -8, -12),
            ("Find cheaper ingredients", "find cheaper suppliers", 900, -5, -3),
        ],
    },
    {
        "title": "New Competitor",
        "story": "A popular coffee chain opens next door. Customers are leaving.",
        "choices": [
            ("Launch a loyalty program", "create a loyalty program", -1200, 7, 15),
            ("Offer big discounts", "offer discounts", -2000, 2, 20),
            ("Improve product quality", "improve product quality", -1500, 10, 12),
        ],
    },
    {
        "title": "Bad Online Reviews",
        "story": "Customers complain about slow service. Your reputation is falling.",
        "choices": [
            ("Train your employees", "train employees", -900, 12, 8),
            ("Hire additional staff", "hire additional staff", -2200, 10, 12),
            ("Run an advertising campaign", "launch social media advertising", -1300, -3, 5),
        ],
    },
    {
        "title": "Rent Increase",
        "story": "Your landlord raises the monthly rent by 25%.",
        "choices": [
            ("Negotiate the rent", "renegotiate rent", 600, 1, 0),
            ("Increase menu prices", "increase menu prices", 1400, -7, -10),
            ("Reduce other expenses", "reduce unnecessary expenses", 900, -2, -2),
        ],
    },
    {
        "title": "Final Sales Crisis",
        "story": "Sales drop before the end of the year. Can you save your coffee shop?",
        "choices": [
            ("Create a marketing campaign", "run a promotional campaign", -1000, 5, 18),
            ("Introduce premium products", "offer premium products", 1700, 3, -4),
            ("Improve customer experience", "improve customer service", -1400, 12, 15),
        ],
    },
]

def reset_game():
    st.session_state.round = 0
    st.session_state.cash = 15000
    st.session_state.reputation = 75
    st.session_state.customers = 120
    st.session_state.history = []

if "round" not in st.session_state:
    reset_game()

st.title("☕ AI Business Crisis Simulator")
st.subheader("Can You Save Your Coffee Shop?")
st.write("Become the CEO, survive five business crises, and learn from AI.")

st.info("Educational simulation: financial effects are predefined. The AI model classifies business strategies, not real financial outcomes.")

c1, c2, c3 = st.columns(3)
c1.metric("Cash Balance", f"$ {st.session_state.cash:,}")
c2.metric("Reputation", f"{st.session_state.reputation}/100")
c3.metric("Customers per Day", st.session_state.customers)

st.progress(st.session_state.round / len(CRISES))
st.caption(f"Completed rounds: {st.session_state.round} / {len(CRISES)}")

if st.session_state.round < len(CRISES):
    crisis = CRISES[st.session_state.round]
    st.header(f"Round {st.session_state.round + 1}: {crisis['title']}")
    st.warning(crisis["story"])

    selected = st.radio(
        "What will you do?",
        range(len(crisis["choices"])),
        format_func=lambda i: crisis["choices"][i][0],
        key=f"choice_{st.session_state.round}",
    )

    if st.button("Make Decision", type="primary"):
        name, strategy_text, cash_change, rep_change, customer_change = crisis["choices"][selected]
        predicted = model.predict(vectorizer.transform([strategy_text]))[0]
        confidence = max(model.predict_proba(vectorizer.transform([strategy_text]))[0])

        st.session_state.cash += cash_change
        st.session_state.reputation = max(0, min(100, st.session_state.reputation + rep_change))
        st.session_state.customers = max(0, st.session_state.customers + customer_change)

        st.session_state.history.append({
            "Round": st.session_state.round + 1,
            "Crisis": crisis["title"],
            "Decision": name,
            "AI Strategy": predicted.replace("_", " ").title(),
            "AI Confidence": round(confidence, 2),
            "Cash": st.session_state.cash,
            "Reputation": st.session_state.reputation,
            "Customers": st.session_state.customers,
        })
        st.session_state.round += 1
        st.rerun()

else:
    st.success("Simulation Completed!")
    if st.session_state.cash > 10000 and st.session_state.reputation >= 65:
        st.balloons()
        st.header("Congratulations! Your coffee shop survived.")
    else:
        st.header("Your business needs a recovery strategy.")

    st.write("### AI Strategy Analysis")
    history = pd.DataFrame(st.session_state.history)
    st.dataframe(history, use_container_width=True, hide_index=True)

    strategy_counts = history["AI Strategy"].value_counts()
    st.bar_chart(strategy_counts)
    st.line_chart(history.set_index("Round")[["Cash"]])
    st.write("Most frequently selected strategy:", strategy_counts.idxmax())

    st.write("### Reflection")
    st.write("Compare your decisions, financial results, and customer satisfaction. Consider which trade-offs helped your business.")

if st.button("Restart Simulation"):
    reset_game()
    st.rerun()
