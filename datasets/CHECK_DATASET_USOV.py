import os
import re
import numpy as np
import pyvista as pv

ROOT = r"C:\Users\M YAQUB\CSU2026\USOV3D"

TRAIN_DIR = os.path.join(ROOT, "Training_Set_2019", "Training_Set_1")
TEST_DIR  = os.path.join(ROOT, "Test_Set_2019", "Test_Set_1")


def read_vtk_volume(path):
    mesh = pv.read(path)

    if len(mesh.point_data.keys()) > 0:
        name = list(mesh.point_data.keys())[0]
        arr = mesh.point_data[name]
    elif len(mesh.cell_data.keys()) > 0:
        name = list(mesh.cell_data.keys())[0]
        arr = mesh.cell_data[name]
    else:
        raise ValueError(f"No scalar data found in: {path}")

    dims = mesh.dimensions

    if len(dims) == 3:
        vol = np.asarray(arr).reshape((dims[2], dims[1], dims[0]))
    else:
        raise ValueError(f"Unexpected VTK dimensions: {dims}")

    return vol


def get_base_cases(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".vtk")]
    base_cases = []

    for f in files:
        if re.match(r"vol\d+\.vtk$", f):
            case_id = f.replace(".vtk", "")
            base_cases.append(case_id)

    return sorted(base_cases, key=lambda x: int(x.replace("vol", "")))


def check_training_cases():
    cases = get_base_cases(TRAIN_DIR)
    print("Total training image volumes:", len(cases))
    print("Cases:", cases)

    for case in cases:
        required = [
            f"{case}.vtk",
            f"{case}_f_r1.vtk",
            f"{case}_f_r2.vtk",
            f"{case}_o_r1.vtk",
            f"{case}_o_r2.vtk",
        ]

        missing = [x for x in required if not os.path.exists(os.path.join(TRAIN_DIR, x))]

        if missing:
            print(f"[Missing] {case}:", missing)
        else:
            print(f"[OK] {case}")

    if len(cases) > 0:
        sample = cases[0]
        img_path = os.path.join(TRAIN_DIR, f"{sample}.vtk")
        img = read_vtk_volume(img_path)

        print("\nSample volume:", sample)
        print("Shape:", img.shape)
        print("Min:", img.min(), "Max:", img.max(), "Mean:", img.mean())


def check_test_cases():
    cases = get_base_cases(TEST_DIR)
    print("\nTotal test image volumes:", len(cases))
    print("Test cases:", cases)

    if len(cases) > 0:
        sample = cases[0]
        img_path = os.path.join(TEST_DIR, f"{sample}.vtk")
        img = read_vtk_volume(img_path)

        print("\nSample test volume:", sample)
        print("Shape:", img.shape)
        print("Min:", img.min(), "Max:", img.max(), "Mean:", img.mean())


check_training_cases()
check_test_cases()