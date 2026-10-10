from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def make_pdf(
    path,
    title,
    dataset_summary,
    model_results,
    fairness_table,
    statistical_text,
    scenario_text="",
    fairness_summary_data=None,
):
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "FairLens audit summary generated from the selected dataset.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    if scenario_text:
        story.append(Paragraph("Experiment Scenario", styles["Heading2"]))
        story.append(Paragraph(scenario_text, styles["BodyText"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Dataset Summary", styles["Heading2"]))
    ds = [
        ["Rows", str(dataset_summary["rows"])],
        ["Columns", str(dataset_summary["columns"])],
        ["Missing Cells", str(dataset_summary["missing_cells"])],
        ["Duplicate Rows", str(dataset_summary["duplicates"])],
    ]
    story.append(Table(
        ds,
        style=TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
        ]),
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Performance", styles["Heading2"]))
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    data = [headers] + [
        [r["Model"]] + [
            f'{r[c]:.3f}' if r[c] == r[c] else "N/A"
            for c in headers[1:]
        ]
        for _, r in model_results.iterrows()
    ]
    story.append(Table(
        data,
        repeatRows=1,
        style=TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]),
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Statistical Analysis", styles["Heading2"]))
    story.append(Paragraph(statistical_text, styles["BodyText"]))
    story.append(Spacer(1, 12))

    if fairness_summary_data:
        story.append(Paragraph("Fairness Summary", styles["Heading2"]))
        fs = [
            ["Metric", "Value"],
            ["Demographic parity difference", f'{fairness_summary_data["Demographic parity difference"]:.3f}'],
            ["Demographic parity ratio", f'{fairness_summary_data["Demographic parity ratio"]:.3f}'],
            ["Equalized odds difference", f'{fairness_summary_data["Equalized odds difference"]:.3f}'],
            ["Equalized odds ratio", f'{fairness_summary_data["Equalized odds ratio"]:.3f}'],
        ]
        story.append(Table(
            fs,
            repeatRows=1,
            style=TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ]),
        ))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Group-wise Fairness Metrics", styles["Heading2"]))
    f_headers = list(fairness_table.columns)
    fdata = [f_headers]
    for _, r in fairness_table.iterrows():
        fdata.append([
            str(r[c]) if c in ["Group", "Count"] else f"{r[c]:.3f}"
            for c in f_headers
        ])
    story.append(Table(
        fdata,
        repeatRows=1,
        style=TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]),
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Interpretation note: FairLens reports observed statistical and model disparities. "
        "Controlled scenarios are experimental and do not represent real-world populations. "
        "Measurements alone do not establish causality, legal discrimination, or unfair intent.",
        styles["BodyText"],
    ))
    doc.build(story)
