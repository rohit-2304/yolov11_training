import json
import glob
import os
from pathlib import Path

# Find the most recent circle_qa_report
report_files = glob.glob("circle_qa_output/circle_qa_report_*.json")
if not report_files:
    print("No circle_qa_report found.")
    exit(1)

latest_report_path = max(report_files, key=os.path.getctime)
print(f"Reading from {latest_report_path}...")

with open(latest_report_path, "r") as f:
    report = json.load(f)

aggregated = {
    "run_timestamp": report.get("run_timestamp"),
    "source_dir": report.get("source_dir"),
    "params_used": report.get("params_used"),
    "summary": {"total": 0, "PASS": 0, "FAIL": 0, "ERROR": 0},
    "products": {}
}

# Group images by product
# Keys in report["images"] look like: "SAP117c2/2/2_snap_03.jpg"
for img_path, img_data in report.get("images", {}).items():
    path_obj = Path(img_path)
    # The parent directory acts as the product identifier (e.g., "SAP117c2/2")
    product_id = str(path_obj.parent)
    
    # We want to extract the result specifically for frame 5 (e.g., "_snap_05.jpg")
    if "_snap_05" in path_obj.name:
        aggregated["products"][product_id] = {
            "representative_image": img_path,
            "status": img_data["status"],
            "defects": img_data.get("defects", []),
            "center": img_data.get("center"),
            "ring_width": img_data.get("ring_width"),
            "panel_saved": img_data.get("panel_saved")
        }
        
        # Update summary based on the representative frame
        aggregated["summary"]["total"] += 1
        aggregated["summary"][img_data["status"]] += 1

output_path = "circle_qa_output/aggregated_circle_qa_report.json"
with open(output_path, "w") as f:
    json.dump(aggregated, f, indent=2)

print(f"Aggregated report saved to {output_path}")
print(f"Total products processed: {aggregated['summary']['total']}")
print(f"PASS: {aggregated['summary']['PASS']}, FAIL: {aggregated['summary']['FAIL']}, ERROR: {aggregated['summary']['ERROR']}")
