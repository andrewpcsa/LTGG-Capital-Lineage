# LTGG Portfolio Capital Lineage — live SharePoint data

This Streamlit app reads the LTGG trade-history workbook directly from SharePoint/OneDrive via Microsoft Graph. The Excel workbook is not stored in GitHub and does not need to be manually uploaded when it changes.

## Repository contents

- `app.py`
- `capital_lineage.py`
- `requirements.txt`
- `.gitignore`
- `.streamlit/secrets.toml.example`

Do **not** add the Excel workbook or a real `.streamlit/secrets.toml` file to GitHub.

## One-time Microsoft setup

The deployed app needs an Entra application/service principal that can read the workbook through Microsoft Graph.

Configure these values in Streamlit Community Cloud under **App settings > Secrets**:

```toml
[microsoft]
tenant_id = "..."
client_id = "..."
client_secret = "..."
```

The app uses the OAuth client-credentials flow and `https://graph.microsoft.com/.default`. Permissions are therefore controlled in the Microsoft Entra application, not in the GitHub code.

For least privilege, ask your Microsoft 365/Entra administrator to grant read access only to the required SharePoint/OneDrive resource where possible, rather than tenant-wide file access.

## Data refresh

The workbook is cached for five minutes to avoid downloading it on every Streamlit widget interaction. The **Refresh data** button clears that cache immediately and downloads the latest saved workbook.

## Deploy

Use `app.py` as the Streamlit entrypoint. There is no `.xlsx` file in this repository.
