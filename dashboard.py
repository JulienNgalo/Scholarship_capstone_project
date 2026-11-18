"""FAO Scholarship Shortlisting | Streamlit prototype.

Run from your existing Scholarship Project directory:
    python -m streamlit run dashboard.py

Does not replace the existing FastAPI service, retrain the model, or adjust scores.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_service import (
    ARTIFACT_PATH, TRAINING_PATH, load_artifact, score_applicants,
    build_shap_explainer, explain_applicant,
)

st.set_page_config(
    page_title="FAO | Scholarship Decision Support",
    page_icon="🎓", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .block-container {max-width: 1480px; padding-top: 1.5rem; padding-bottom: 2rem;}
  h1, h2, h3 {letter-spacing: -.02em;}
  div[data-testid="stMetric"] {padding: 12px 16px; border: 1px solid #D9E2EA;
       border-radius: 10px; background: rgba(119,142,157,.055);}
  div[data-testid="stMetricLabel"] {font-weight: 600;}
  div[data-testid="stAlert"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved Gradient Boosting model …")
def get_model():
    return load_artifact(ARTIFACT_PATH)


@st.cache_resource(show_spinner="Preparing SHAP explanations …")
def get_explainer():
    return build_shap_explainer(get_model(), TRAINING_PATH)


def probability_chart(baseline: float, effects: pd.DataFrame, probability: float):
    """Display signed SHAP contributions for the ten original model features."""
    ranked = effects.sort_values("Contribution (pp)").copy()
    fig = go.Figure(go.Bar(
        x=ranked["Contribution (pp)"], y=ranked["Feature"], orientation="h",
        marker_color=["#0f766e" if x >= 0 else "#c46c58"
                      for x in ranked["Contribution (pp)"]],
        customdata=ranked["Applicant value"],
        hovertemplate="%{y}<br>Value: %{customdata}<br>Impact: %{x:+.2f} percentage points<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#64748b", line_width=1)
    fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=5, r=5, t=16, b=15),
        xaxis_title="Effect on predicted approval (percentage points)",
        yaxis_title=None, font=dict(size=12), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"SHAP baseline: {baseline:.1%}  •  Combined prediction: {probability:.1%}. "
        "Positive bars increase the predicted probability relative to the background; "
        "negative bars decrease it. These are explanations, not eligibility rules."
    )


def explain_selected(features: pd.DataFrame, artifact: dict, probability: float):
    if not TRAINING_PATH.is_file():
        st.info("To enable SHAP, place artifacts/training_data.csv in this project. "
                "Predictions will still work without it.")
        return
    try:
        explainer, groups = get_explainer()
        baseline, effects = explain_applicant(features, artifact, explainer, groups)
    except Exception as exc:
        st.error(f"SHAP explanation unavailable: {exc}")
        return
    probability_chart(baseline, effects, probability)
    with st.expander("Feature contributions and values", expanded=False):
        st.dataframe(effects, hide_index=True, use_container_width=True)
    with st.expander("All 10 engineered model features"):
        st.dataframe(features.T.rename(columns={features.index[0]: "Value"}),
                     use_container_width=True)


def result_metrics(scores: pd.DataFrame, threshold: float):
    a, b, c, d = st.columns(4)
    approved = int((scores["Predicted status"] == "Approved").sum())
    a.metric("Scored applicants", f"{len(scores):,}")
    b.metric("Predicted Approved", f"{approved:,}")
    c.metric("Predicted Not Approved", f"{len(scores) - approved:,}")
    d.metric("Model threshold", f"{threshold:.0%}")


def display_table(scores: pd.DataFrame, key: str, selection_callback):
    columns = ["Student ID", "Approval %", "Predicted status", "Mean Grade", "County", "Course"]
    display = scores.copy()
    display["Approval %"] = display["Approval probability"] * 100.0
    event = st.dataframe(
        display[columns].reset_index(drop=True),
        key=key, hide_index=True, use_container_width=True, height=300,
        on_select=selection_callback, selection_mode="single-row",
        column_config={"Approval %": st.column_config.NumberColumn(
            "Approval %", format="%.2f%%")},
    )
    return event


def csv_download(scores: pd.DataFrame) -> bytes:
    export = scores.drop(columns=["_row_number"], errors="ignore").copy()
    export["Approval probability"] = (100 * export["Approval probability"]).round(2)
    export["Not approved probability"] = (100 * export["Not approved probability"]).round(2)
    export = export.rename(columns={"Approval probability": "Approval probability (%)",
                                    "Not approved probability": "Not approved probability (%)"})
    return export.to_csv(index=False).encode("utf-8-sig")


def individual_page(artifact):
    st.subheader("Individual applicant assessment")
    st.write("Enter an applicant's details. The model uses the same feature engineering as your API.")
    left, right = st.columns([0.94, 1.06], gap="large")
    county_options = sorted(k.title() for k in artifact["county_rates"].keys())
    course_options = sorted(k.title() for k in artifact["faculty_schools"].keys())
    with left:
        with st.form("individual_form"):
            student_id = st.text_input("Student ID (optional)", placeholder="e.g. FAO-TEST-001")
            c1, c2 = st.columns(2)
            with c1:
                mean_grade = st.selectbox("Mean grade", ["A", "A-", "B+", "B", "B-", "C+",
                                                         "C", "C-", "D+", "D", "D-", "E",
                                                         "A LEVELS", "DIPLOMA"], index=2)
                county = st.selectbox("County", county_options,
                                      index=county_options.index("Nairobi") if "Nairobi" in county_options else 0)
                parent_status = st.selectbox("Parent status", ["Both Parents", "Mother Deceased",
                                                                     "Father Deceased", "Orphan"])
            with c2:
                o_year = st.number_input("O-level completion year", min_value=1950,
                                         max_value=2100, value=2023, step=1)
                applied = st.date_input("Date applied", value=date.today(),
                                        min_value=date(1950, 1, 1), max_value=date(2100, 12, 31))
            high_school = st.text_input("High school", value="Alliance High School")
            course = st.selectbox("University course", course_options,
                                  index=course_options.index("Bachelor Of Commerce")
                                  if "Bachelor Of Commerce" in course_options else 0)
            submitted = st.form_submit_button("Calculate shortlisting probability", type="primary",
                                               use_container_width=True)
        if submitted:
            raw = pd.DataFrame([{
                "student_id": student_id or "Applicant 1", "mean_grade": mean_grade,
                "o_level_year": o_year, "county": county,
                "date_applied": applied.isoformat(), "high_school": high_school,
                "parent_status": parent_status, "course": course,
            }])
            try:
                score = score_applicants(raw, artifact)
                if not score.errors.empty:
                    st.error(score.errors.iloc[0]["Error"])
                    st.session_state.pop("single_result", None)
                else:
                    st.session_state["single_result"] = score
            except Exception as exc:
                st.error(f"Unable to process applicant: {exc}")
                st.session_state.pop("single_result", None)
    with right:
        score = st.session_state.get("single_result")
        if score is None:
            st.info("Submit the form to view the probability, provisional prediction, and SHAP explanation.")
            return
        record = score.scores.iloc[0]
        st.markdown("#### Model result")
        a, b = st.columns(2)
        a.metric("Predicted approval probability", f"{record['Approval probability']:.2%}")
        b.metric("Predicted status", record["Predicted status"])
        st.progress(float(record["Approval probability"]))
        st.caption("Not-approved probability: "
                   f"{record['Not approved probability']:.2%}  •  "
                   "Provisional shortlisting prediction, not an award decision.")
        if record["Check notes"]:
            st.warning(record["Check notes"])
        st.markdown("#### Why did the model produce this result?")
        explain_selected(score.features.iloc[[0]], artifact,
                         float(record["Approval probability"]))


def batch_page(artifact):
    st.subheader("Batch applicant screening")
    st.write("Upload a CSV with one student per row. Ranked results are predictions for human review, not final awards.")
    template = pd.DataFrame([
        {"student_id": "EXAMPLE-001", "mean_grade": "B+", "o_level_year": 2023,
         "county": "Nairobi", "date_applied": "2026-11-18",
         "high_school": "Alliance High School", "parent_status": "Both Parents",
         "course": "Bachelor of Commerce"},
        {"student_id": "EXAMPLE-002", "mean_grade": "C", "o_level_year": 2019,
         "county": "Wajir", "date_applied": "2026-11-18",
         "high_school": "British International School", "parent_status": "Both Parents",
         "course": "Bachelor of Arts International Studies"},
    ])
    st.download_button("Download sample CSV template", template.to_csv(index=False).encode("utf-8-sig"),
                       file_name="fao_batch_template.csv", mime="text/csv")
    file = st.file_uploader("Upload applicant CSV", type=["csv"], accept_multiple_files=False,
                            help="Required: mean_grade, o_level_year, county, date_applied, "
                                 "high_school, parent_status, course. Optional: student_id.")
    if file is None:
        st.info("Upload a CSV to see the ranked shortlist on the left and the selected student's "
                "SHAP explanation on the right.")
        return
    raw_bytes = file.getvalue()
    if len(raw_bytes) > 2_000_000:
        st.error("For this prototype, limit CSV files to 2 MB.")
        return
    try:
        raw = pd.read_csv(BytesIO(raw_bytes), dtype={"student_id": str, "ID": str})
        if not 1 <= len(raw) <= 1000:
            st.error("For this prototype, upload 1–1,000 applicants per file.")
            return
        batch = score_applicants(raw, artifact)
    except Exception as exc:
        st.error(f"Could not process the CSV: {exc}")
        return
    if not batch.errors.empty:
        st.warning(f"{len(batch.errors)} invalid rows were not scored. Review them below.")
        with st.expander("Rows requiring correction", expanded=True):
            st.dataframe(batch.errors, hide_index=True, use_container_width=True)
            st.download_button("Download invalid rows report", batch.errors.to_csv(index=False),
                               file_name="fao_invalid_rows.csv", mime="text/csv")
    if batch.scores.empty:
        st.error("No valid applicant rows to score.")
        return
    result_metrics(batch.scores, float(artifact.get("threshold", 0.5)))
    scores = batch.scores.sort_values("Approval probability", ascending=False, kind="stable")
    shortlisted = scores[scores["Predicted status"] == "Approved"].reset_index(drop=True)
    not_shortlisted = scores[scores["Predicted status"] == "Not Approved"].reset_index(drop=True)

    main, side = st.columns([1.24, 1], gap="large")
    with main:
        st.markdown("#### Ranked predicted shortlist")
        st.caption("Highest probability first. Click a row to explain that student's prediction.")
        if shortlisted.empty:
            st.info("No applicants meet the model's current threshold.")
        else:
            def shortlist_clicked():
                selected = st.session_state["shortlist_table"]["selection"]["rows"]
                if selected:
                    st.session_state["selected_csv_row"] = int(shortlisted.iloc[selected[0]]["_row_number"])
            display_table(shortlisted, "shortlist_table", shortlist_clicked)
            st.download_button("Export predicted shortlist", csv_download(shortlisted),
                               file_name="fao_predicted_shortlist.csv", mime="text/csv")
        st.download_button("Export all scored applicants", csv_download(scores),
                           file_name="fao_all_predictions.csv", mime="text/csv")
    with side:
        st.markdown("#### Selected applicant · SHAP explanation")
        valid_rows = scores["_row_number"].astype(int).tolist()
        selected_row = st.session_state.get("selected_csv_row")
        if selected_row not in valid_rows:
            selected_row = valid_rows[0]
        selected = scores[scores["_row_number"] == selected_row].iloc[0]
        st.caption(f"Student: {selected['Student ID']} · original CSV row {selected_row}")
        st.metric("Predicted approval probability", f"{selected['Approval probability']:.2%}")
        st.caption(f"Classification: {selected['Predicted status']} · "
                   "This is a model output, not an award decision.")
        if selected["Check notes"]:
            st.warning(selected["Check notes"])
        explain_selected(batch.features.loc[[selected_row]], artifact,
                         float(selected["Approval probability"]))
        st.divider()
        st.markdown("#### Ranked predicted Not Approved")
        st.caption("Highest probability among non-shortlisted first. Click to inspect.")
        if not_shortlisted.empty:
            st.info("No applicants below the threshold.")
        else:
            def not_shortlist_clicked():
                selected_rows = st.session_state["not_shortlist_table"]["selection"]["rows"]
                if selected_rows:
                    st.session_state["selected_csv_row"] = int(not_shortlisted.iloc[selected_rows[0]]["_row_number"])
            display_table(not_shortlisted, "not_shortlist_table", not_shortlist_clicked)
            st.download_button("Export predicted Not Approved", csv_download(not_shortlisted),
                               file_name="fao_predicted_not_shortlisted.csv", mime="text/csv")


st.title("FAO  |  Scholarship Decision Support")
st.caption("Historical approval model · Gradient Boosting · Research prototype")
st.info("Outputs are historical approval predictions, **not scholarship eligibility decisions**. "
        "Staff should review the evidence and policy criteria before shortlisting or awards.",
        icon="ℹ️")
try:
    artifact = get_model()
except Exception as error:
    st.error(f"Cannot load model: {error}")
    st.stop()

with st.sidebar:
    st.markdown("### Model information")
    st.write(f"**Model:** {artifact.get('model_name', 'Unknown')}")
    st.write(f"**Decision threshold:** {artifact.get('threshold', 0.5):.0%}")
    st.write(f"**Original features:** {len(artifact['feature_columns'])}")
    st.caption("Uses the same src.data_pipeline.build_features() as FastAPI.")
    st.markdown("---")
    st.warning("Local prototype only. Restrict access before using real applicant records; "
               "do not deploy this unauthenticated interface publicly.")

tab_individual, tab_batch = st.tabs(["Individual assessment", "Batch shortlisting"])
with tab_individual:
    individual_page(artifact)
with tab_batch:
    batch_page(artifact)
