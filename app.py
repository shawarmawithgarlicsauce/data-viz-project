"""Predictive maintenance dashboard on the UCI AI4I 2020 dataset."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA = "data/ai4i2020.csv"
FEATURES = ["Type_code", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "Power [W]", "Temp diff [K]"]
FAILURE_TYPES = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain", "RNF": "Random"}


@st.cache_data
def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["Type_code"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


@st.cache_resource
def train(df):
    # Only sensor features: the failure-mode columns (TWF..RNF) would leak the label.
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df["Machine failure"], test_size=0.2,
                                              stratify=df["Machine failure"], random_state=42)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                   random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    return model, X_te, y_te


def scores(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))


def cost(s, c_fail, c_inspect):
    # Missed failures cost a breakdown; every alarm (true or false) costs a planned inspection.
    return s["fn"] * c_fail + (s["tp"] + s["fp"]) * c_inspect


st.set_page_config(page_title="FailSafe – Predictive Maintenance", layout="wide")
df = load()
model, X_te, y_te = train(df)
proba = model.predict_proba(X_te)[:, 1]

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Alert threshold (failure probability)", 0.05, 0.95, 0.30, 0.05)
    c_fail = st.number_input("Cost of one unplanned breakdown (RM)", 0, 1_000_000, 20_000, 1_000)
    c_inspect = st.number_input("Cost of one planned inspection (RM)", 0, 100_000, 1_500, 100)
    st.caption("Costs are assumptions — set them to match your plant.")

st.title("FailSafe: Predictive Maintenance Dashboard")
tab_over, tab_explore, tab_model, tab_cost, tab_predict = st.tabs(
    ["Overview", "Explore sensors", "Model performance", "Cost impact", "Predict a machine"])

with tab_over:
    c = st.columns(4)
    c[0].metric("Machine records", f"{len(df):,}")
    c[1].metric("Failures", int(df["Machine failure"].sum()))
    c[2].metric("Failure rate", f"{df['Machine failure'].mean():.2%}")
    c[3].metric("Product types", "L / M / H")
    left, right = st.columns(2)
    counts = df[list(FAILURE_TYPES)].sum().rename(index=FAILURE_TYPES).sort_values()
    left.plotly_chart(px.bar(counts, orientation="h", title="Failures by mode",
                             labels={"value": "Count", "index": ""}).update_layout(showlegend=False),
                      width="stretch")
    by_type = df.groupby("Type")["Machine failure"].mean().reindex(["L", "M", "H"]) * 100
    right.plotly_chart(px.bar(by_type, title="Failure rate by product quality type (%)",
                              labels={"value": "Failure rate (%)", "Type": "Type"}).update_layout(showlegend=False),
                       width="stretch")

with tab_explore:
    num = [f for f in FEATURES if f != "Type_code"]
    x = st.selectbox("X axis", num, index=num.index("Rotational speed [rpm]"))
    y = st.selectbox("Y axis", num, index=num.index("Torque [Nm]"))
    plot_df = df.assign(Status=df["Machine failure"].map({0: "Normal", 1: "Failure"})).sort_values("Machine failure")
    st.plotly_chart(px.scatter(plot_df, x=x, y=y, color="Status", opacity=0.6,
                               color_discrete_map={"Normal": "#9aa5b1", "Failure": "#d62728"}),
                    width="stretch")
    st.plotly_chart(px.histogram(plot_df, x=x, color="Status", barmode="overlay", histnorm="percent",
                                 color_discrete_map={"Normal": "#9aa5b1", "Failure": "#d62728"}),
                    width="stretch")

pred = (proba >= threshold).astype(int)
m = scores(y_te, pred)
# Benchmark: today's fixed-interval practice — replace the tool once wear passes 200 min.
base = scores(y_te, (X_te["Tool wear [min]"] > 200).astype(int))

with tab_model:
    st.caption(f"Random Forest tested on a held-out 20% split ({len(y_te):,} records, {int(y_te.sum())} failures).")
    c = st.columns(4)
    c[0].metric("Recall (failures caught)", f"{m['recall']:.1%}", f"{m['recall'] - base['recall']:+.1%} vs baseline")
    c[1].metric("Precision", f"{m['precision']:.1%}", f"{m['precision'] - base['precision']:+.1%} vs baseline")
    c[2].metric("F1 score", f"{m['f1']:.2f}", f"{m['f1'] - base['f1']:+.2f} vs baseline")
    c[3].metric("ROC-AUC", f"{roc_auc_score(y_te, proba):.3f}")
    left, right = st.columns(2)
    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                      index=["Actual normal", "Actual failure"], columns=["Predicted normal", "Predicted failure"])
    left.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix"),
                      width="stretch")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    right.plotly_chart(px.bar(imp, orientation="h", title="Feature importance",
                              labels={"value": "Importance", "index": ""}).update_layout(showlegend=False),
                       width="stretch")
    st.dataframe(pd.DataFrame({"Fixed-interval baseline": base, "FailSafe model": m}).T
                 [["recall", "precision", "f1", "tp", "fp", "fn"]].style.format(precision=3))

with tab_cost:
    no_plan = int(y_te.sum()) * c_fail
    rows = pd.DataFrame({"Strategy": ["Run to failure", "Fixed-interval baseline", "FailSafe model"],
                         "Cost (RM)": [no_plan, cost(base, c_fail, c_inspect), cost(m, c_fail, c_inspect)]})
    saving = 1 - rows["Cost (RM)"].iloc[2] / no_plan
    c = st.columns(3)
    c[0].metric("Breakdowns avoided", int(m["tp"]), f"of {int(y_te.sum())}")
    c[1].metric("Maintenance cost saved vs run-to-failure", f"{saving:.0%}")
    c[2].metric("False alarms", int(m["fp"]))
    st.plotly_chart(px.bar(rows, x="Strategy", y="Cost (RM)", text_auto=",.0f",
                           title=f"Maintenance cost on the test set ({len(y_te):,} records)"),
                    width="stretch")

with tab_predict:
    st.write("Enter current sensor readings to get a failure risk and a recommendation.")
    c = st.columns(3)
    typ = c[0].selectbox("Product type", ["L", "M", "H"])
    air = c[0].slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    proc = c[1].slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
    rpm = c[1].slider("Rotational speed [rpm]", 1150, 2900, 1500)
    torque = c[2].slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
    wear = c[2].slider("Tool wear [min]", 0, 255, 100)
    row = pd.DataFrame([[{"L": 0, "M": 1, "H": 2}[typ], air, proc, rpm, torque, wear,
                         torque * rpm * 2 * np.pi / 60, proc - air]], columns=FEATURES)
    p = model.predict_proba(row)[0, 1]
    st.metric("Failure probability", f"{p:.1%}")
    if p >= threshold:
        st.error("High risk: schedule maintenance before the next shift.")
    elif p >= threshold / 2:
        st.warning("Elevated risk: monitor closely and inspect at the next planned stop.")
    else:
        st.success("Normal: no action needed.")
