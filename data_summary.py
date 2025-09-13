import json
import pandas as pd
from collections import Counter
from datetime import datetime

def load_data():
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    # Handle both possible data structures from API
    if 'output' in data:
        policies = data['output'][0]['data']
    else:
        policies = data['data']
    
    return policies

def analyze_data(policies):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(policies)
    
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
                "null_count": len(df)
            }
            continue
            
        field_type = str(values.dtype)
        
        if field_type in ['object', 'string']:
            # For string fields, show unique values if few, otherwise show count
            unique_vals = values.unique()
            if len(unique_vals) < 20:  # Only show if manageable number of unique values
                val_summary = list(unique_vals)
            else:
                val_summary = f"{len(unique_vals)} unique values"
                
            # Show top 5 most common values and their counts
            value_counts = values.value_counts().head(5).to_dict()
                
            summary["field_value_summary"][column] = {
                "type": field_type,
                "unique_values": val_summary,
                "null_count": df[column].isna().sum(),
                "top_values": value_counts
            }
        else:
            # For numeric fields, show range and distribution
            summary["field_value_summary"][column] = {
                "type": field_type,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "null_count": df[column].isna().sum()
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
    policies = load_data()
    summary = analyze_data(policies)
    summary_text = format_summary(summary)
    
    # Write summary to file
    with open("data_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    # Also print to console
    print(summary_text)

if __name__ == "__main__":
    main()
