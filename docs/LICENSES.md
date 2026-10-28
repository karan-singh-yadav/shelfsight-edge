# Licenses and redistribution (not legal advice: read the originals)

| Item | License / terms | Consequence for this repo |
|---|---|---|
| This project's code | MIT | free to reuse |
| Ultralytics (library, YOLOv8 weights) | AGPL-3.0 (or paid Enterprise licence) | Not vendored. The Docker image deliberately excludes it (ONNX Runtime only). Distributing a combined work that includes it can trigger AGPL obligations; check before publishing images or hosted services. |
| SKU-110K dataset | Academic, non-commercial use only (per its README) | Never commit or redistribute the data. Think carefully before publishing weights trained on it. |
| ONNX Runtime, ONNX, NumPy, pandas, OpenCV, SciPy, Streamlit, Plotly | permissive (MIT/BSD/Apache-2.0 families) | verify versions you ship |
| Your custom dataset images | whatever each source allows | record source + licence per image in `sources.csv`; `verify-splits` warns on `unknown` |

Suggested approach for a public portfolio: publish code, configs, metrics and screenshots; keep datasets out; publish weights only if every upstream licence allows it, otherwise document how to reproduce them.
