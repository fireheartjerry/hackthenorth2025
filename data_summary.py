import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from utils_data import load_df


def load_data(path: str = "data.json") -> pd.DataFrame:
    """Load and normalize policy data using the shared helper."""
    return load_df(path)


def analyze_data(df: pd.DataFrame):
    """Generate a high level summary of the provided DataFrame."""
    # Ensure we always operate on a DataFrame
    if df is None:
        df = pd.DataFrame()
    
    summary = {
        "total_records": len(df),
        "fields_present": list(df.columns),
        "field_value_summary": {}
    }
    
    # Analyze each field
    for column in df.columns:
        values = df[column].dropna()

        if len(values) == 0:
            summary["field_value_summary"][column] = {
                "type": "empty",
                "unique_values": [],
                "null_count": len(df),
            }
            continue

        if is_numeric_dtype(values):
            summary["field_value_summary"][column] = {
                "type": "number",
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "null_count": df[column].isna().sum(),
            }
        elif is_datetime64_any_dtype(values):
            summary["field_value_summary"][column] = {
                "type": "datetime",
                "min": values.min().isoformat(),
                "max": values.max().isoformat(),
                "null_count": df[column].isna().sum(),
            }
        else:
            # For string-like fields, show unique values if few, otherwise show count
            unique_vals = values.astype(str).unique()
            if len(unique_vals) < 20:
                val_summary = list(unique_vals)
            else:
                val_summary = f"{len(unique_vals)} unique values"

            value_counts = values.astype(str).value_counts().head(5).to_dict()

            summary["field_value_summary"][column] = {
                "type": "string",
                "unique_values": val_summary,
                "null_count": df[column].isna().sum(),
                "top_values": value_counts,
            }
    
    return summary

def format_summary(summary):
    lines = []
    lines.append("DATA SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Records: {summary['total_records']}")
    lines.append("\nFields Present:")
    for field in summary['fields_present']:
        lines.append(f"- {field}")
    
    lines.append("\nField Analysis:")
    lines.append("=" * 80)
    for field, analysis in summary['field_value_summary'].items():
        lines.append(f"\n{field}:")
        for key, value in analysis.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)

def main():
    df = load_data()
    summary = analyze_data(df)
    summary_text = format_summary(summary)
    
    # Write summary to file
    with open("data_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    # Also print to console
    print(summary_text)

if __name__ == "__main__":
    main()
