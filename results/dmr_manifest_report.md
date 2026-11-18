# DMR-IR manifest

Source: `data/DMR-IR/metadata.csv`

Start: **6878** images, **268** patients

- excluded `volunteer`: **32** patients -> ['0004', '0005', '0006', '0007', '0008', '0012', '0013', '0015', '0016', '0017', '0031', '0038'] ...
- excluded `duplicate_patient`: **0** patients

## Final cohort

- patients: **236**
- images: **6051**
- labels: {0: 140, 1: 96}  (0 = benign, 1 = malignant)
- roles: {'Patient': 236}

```
view_group
frontal      4620
lateral90     917
lateral45     514
```

`view_side_inferred` follows Figure 7 of Silva et al. (2014) and is UNVERIFIED;
confirm with the mirror-correlation check in data/README.md before using laterality.

## Next

```bash
python scripts/02_create_patient_splits.py \
    --manifest results/dmr_patient_manifest.csv \
    --output   results/dmr_patient_splits.csv
```
