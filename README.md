# Portfolio Capital Lineage dashboard

A Streamlit dashboard that traces capital released by portfolio sales through subsequent purchases, additions and later reallocations. Each sell-A / buy-B decision is scored against the counterfactual of retaining A. Positive relative-performance decisions curve upward; negative decisions curve downward.

## Repository contents

Keep these files together in the root of the GitHub repository:

```text
capital-lineage-dashboard/
├── app.py
├── capital_lineage.py
├── requirements.txt
├── README.md
├── .gitignore
└── LTGG Full Trade History for New Sankey.xlsx
```

The app automatically reads `LTGG Full Trade History for New Sankey.xlsx` from the repository. There is no workbook upload step.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the app from the repository folder:

```bash
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and add the files above.
2. Because the workbook contains portfolio trade history, use a **private repository** unless the data is cleared for public release.
3. In Streamlit Community Cloud, create a new app from the GitHub repository.
4. Select the branch containing the files and use `app.py` as the entrypoint.
5. Deploy. The workbook is read directly from the deployed repository.

When the workbook changes, replace the `.xlsx` file in GitHub and push/commit the update. Streamlit Community Cloud will redeploy the repository version.

## What the prototype does

- Matches sale proceeds to the closest subsequent purchases, prorating multiple sales on the same date.
- Propagates original-sale ancestry through additions, partial sales and complete sales.
- Uses the workbook's total-return index history to compare each sell-A / buy-B decision with the counterfactual of retaining A.
- Draws a left-to-right lineage graph with edge width proportional to selected-root capital flow.
- Curves positive relative-performance decisions upward and negative decisions downward.
- Exposes every allocation in an audit table and allows CSV export.

See the in-app assumptions panel before using the output as formal attribution. `% Portfolio Order` is a practical lineage unit, not literal cash carried unchanged across time.
