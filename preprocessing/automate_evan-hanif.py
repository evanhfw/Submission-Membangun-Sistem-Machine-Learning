import re

from typing import Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


def clip_extreme_values(
    s: pd.Series, rules: Tuple[Optional[int | float], Optional[int | float]]
) -> pd.Series:
    """
    Convert values outside [lo, hi] to NaN for a single Series.

    Parameters
    ----------
    s : pd.Series
        Single column data.
    rules : tuple (lo, hi)
        Lower and upper bounds. Use None to skip either one.
        Examples: (0, 10), (None, 100), (18, None)
    """
    print(f"Processing extreme values in column '{s.name}' with bounds {rules}")
    s = s.copy()
    lo, hi = rules

    original_count = s.count()

    if lo is not None:
        below_count = (s < lo).sum()
        s.loc[s < lo] = np.nan
        if below_count > 0:
            print(f"  - Converting {below_count} extreme values below {lo} to NaN")

    if hi is not None:
        above_count = (s > hi).sum()
        s.loc[s > hi] = np.nan
        if above_count > 0:
            print(f"  - Converting {above_count} extreme values above {hi} to NaN")

    final_count = s.count()
    removed_count = original_count - final_count

    if removed_count > 0:
        print(f"  - Total {removed_count} unreasonable extreme values converted to NaN")
    else:
        print("  - No unreasonable extreme values found")

    return s


def clean_missing_values(s: pd.Series) -> pd.Series:
    """
    Clean and standardize missing value representations in a pandas Series.

    This function replaces various representations of missing values with np.nan
    to ensure consistent handling of missing data throughout the dataset.

    Args:
        s (pd.Series): Input series containing potential missing value representations

    Returns:
        pd.Series: Series with standardized missing values (np.nan)

    Note:
        Replaces the following patterns with np.nan:
        - "_______" (seven underscores)
        - "_" (single underscore)
        - "!@9#%8" (special character pattern)
        - "NM" (Not Measured/Not Available)
    """
    print(f"Cleaning missing values in column '{s.name}'")
    return (
        s.replace("_______", np.nan)
        .replace("_", np.nan)
        .replace("!@9#%8", np.nan)
        .replace("NM", np.nan)
    )


def convert_to_number(s: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric values by removing non-numeric characters.

    Args:
        s (pd.Series): Input series to convert to numeric values

    Returns:
        pd.Series: Series with numeric values, non-convertible values become NaN
    """
    print(f"Processing column '{s.name}' to convert to numeric")
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.]", "", regex=True).replace("", None),
        errors="coerce",
    )


def convert_duration_to_months(s: pd.Series) -> pd.Series:
    """
    Convert duration strings to total months.

    Supports various formats including:
    - "2 years 3 months", "2 yrs 3 mos", "2 tahun 3 bulan"
    - "5 years", "3 months", "2 thn", "6 bln"
    - Case-insensitive matching

    Args:
        s (pd.Series): Input series containing duration strings

    Returns:
        pd.Series: Series with duration converted to total months
    """
    print(f"Processing column '{s.name}' for duration to months conversion")

    # Regex fleksibel (case-insensitive)
    pat = (
        r"(?i)^\s*"
        r"(?:(\d+)\s*(?:years?|yrs?|yr|tahun|thn))?\s*"
        r"(?:and|&|,)?\s*"
        r"(?:(\d+)\s*(?:months?|mos?|mo|bulan|bln))?\s*$"
    )

    ext = s.astype(str).str.extract(pat)  # dua kolom: [years, months]

    yrs = pd.to_numeric(ext[0], errors="coerce")
    mos = pd.to_numeric(ext[1], errors="coerce")

    total_months = yrs.fillna(0) * 12 + mos.fillna(0)

    print(f"Successfully converted {len(total_months)} rows of duration to months")

    return total_months


def process_type_of_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Type_of_Loan column by converting it to multiple binary columns.

    This function converts the Type_of_Loan column which contains multiple loan types
    separated by commas, ampersands, or 'and' into separate binary columns for each
    loan type category.

    Args:
        df (pd.DataFrame): Input dataframe containing Type_of_Loan column

    Returns:
        pd.DataFrame: Dataframe with Type_of_Loan column replaced by multiple binary columns
    """
    print("Processing 'Type_of_Loan' column for conversion to multiple binary columns")

    COL = "Type_of_Loan"

    # 1) Normalize separator
    SEP_RE = re.compile(r"\s*(?:,|&|and)\s*", flags=re.I)

    def split_tokens(x):
        if pd.isna(x):
            return []
        return [
            t.strip(" ,").strip() for t in SEP_RE.split(str(x)) if t.strip(" ,").strip()
        ]

    # 2) Count raw tokens (for audit)
    raw_lists = df[COL].apply(split_tokens)

    # ---- Audit Unknown causes ----
    # a) only Not Specified / empty
    only_ns = raw_lists.apply(
        lambda lst: len(lst) > 0
        and all(s.lower() in {"not specified", "n/a", "na"} for s in lst)
    )
    empty_or_ns = df[COL].isna() | only_ns
    print(f"Only 'Not Specified' / NaN: {int(empty_or_ns.sum())}")

    # 3) Pattern dictionary → canonical labels (with common aliases)
    PATTERNS = [
        (r"credit[\s-]*builder", "Credit-Builder Loan"),
        (r"\bpersonal\b|\bcons?umer\b|\bline of credit\b", "Personal Loan"),
        (r"debt\s*consol|consolidation", "Debt Consolidation Loan"),
        (r"\bstudent\b|education", "Student Loan"),
        (r"\bpay\s*day\b|\bcash\s*advance\b", "Payday Loan"),
        (r"\bmort(gage)?\b|\bhome\s*loan\b", "Mortgage Loan"),
        (r"\bauto\b|\bcar\b|\bvehicle\b", "Auto Loan"),
        (r"home\s*equity", "Home Equity Loan"),
    ]
    CANON = [lab for _, lab in PATTERNS]
    NOISE = {"not specified", "n/a", "na", "", "-", "--"}

    def map_row(lst):
        mapped, had_unknown = set(), False
        if not lst:
            return [], True
        for tok in lst:
            low = tok.lower()
            if low in NOISE:
                continue
            hit = None
            for pat, canon in PATTERNS:
                if re.search(pat, low, flags=re.I):
                    hit = canon
                    break
            if hit is None:
                had_unknown = True
            else:
                mapped.add(hit)
        if not mapped:
            had_unknown = True
        return sorted(mapped), had_unknown

    mapped = raw_lists.apply(map_row)
    loan_lists = mapped.apply(lambda x: x[0])
    unknown_flag = mapped.apply(lambda x: x[1]).astype(int)

    # 4) Build multi-hot for 8 labels
    def slug(s):
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    created = []
    for lab in CANON:
        coln = f"Loan_{slug(lab)}"
        df[coln] = loan_lists.apply(lambda L: int(lab in L))
        created.append(coln)

    df["Loan_Unknown"] = unknown_flag

    print(f"Created columns: {created + ['Loan_Unknown']}")
    print(f"Unknown rows: {int(df['Loan_Unknown'].sum())}")

    # 5) Look at sample rows that are still unknown but not empty/Not Specified
    mask_unknown_nonempty = (df["Loan_Unknown"] == 1) & (~empty_or_ns)
    sample = df.loc[mask_unknown_nonempty, COL].head(20)
    print(f"Sample unknown non-empty: {len(sample)} rows")

    # Drop original column
    df = df.drop(COL, axis=1)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process data by applying extreme value clipping,
    data type conversion, and missing value cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be cleaned
    target_column : str
        Target column name (for reference)
    save_path : str
        Path to save preprocessing results
    file_path : str
        Original file path (for reference)

    Returns
    -------
    pd.DataFrame
        Cleaned and processed DataFrame

    Notes
    -----
    This function performs:
    1. Conversion and clipping of numeric values with reasonable bounds
    2. Duration conversion to months
    3. Cleaning of missing categorical values
    4. Processing of Type_of_Loan column to multi-hot encoding
    """
    print("=" * 60)
    print("STARTING DATA CLEANING PROCESS")
    print("=" * 60)

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    # Convert and clip numerical columns in one step
    print("\n📊 PROCESSING NUMERIC COLUMNS:")
    print("-" * 40)

    df["Age"] = clip_extreme_values(convert_to_number(df["Age"]), (1, 70))
    df["Annual_Income"] = clip_extreme_values(
        convert_to_number(df["Annual_Income"]), (1, 2e5)
    )
    df["Num_Bank_Accounts"] = clip_extreme_values(df["Num_Bank_Accounts"], (1, 10))
    df["Num_Credit_Card"] = clip_extreme_values(df["Num_Credit_Card"], (1, 10))
    df["Interest_Rate"] = clip_extreme_values(df["Interest_Rate"], (0, 100))
    df["Num_of_Loan"] = clip_extreme_values(
        convert_to_number(df["Num_of_Loan"]), (1, 10)
    )
    df["Num_of_Delayed_Payment"] = clip_extreme_values(
        convert_to_number(df["Num_of_Delayed_Payment"]), (1, 30)
    )
    df["Num_Credit_Inquiries"] = clip_extreme_values(
        df["Num_Credit_Inquiries"], (1, 30)
    )
    df["Total_EMI_per_month"] = clip_extreme_values(df["Total_EMI_per_month"], (0, 900))
    df["Amount_invested_monthly"] = clip_extreme_values(
        convert_to_number(df["Amount_invested_monthly"]), (0, 1500)
    )
    df["Monthly_Balance"] = clip_extreme_values(
        convert_to_number(df["Monthly_Balance"]), (0, 2000)
    )
    df["Changed_Credit_Limit"] = clip_extreme_values(
        convert_to_number(df["Changed_Credit_Limit"]), (1, 10)
    )

    print("\n💰 PROCESSING COLUMNS WITHOUT CLIPPING:")
    print("-" * 40)

    # Convert remaining numerical columns without clipping
    print("Processing 'Outstanding_Debt' column without clipping")
    df["Outstanding_Debt"] = convert_to_number(df["Outstanding_Debt"])

    print("\n📅 DURATION TO MONTHS CONVERSION:")
    print("-" * 40)

    # Convert credit history age to months and drop original column
    print("Converting 'Credit_History_Age' to months and dropping original column")
    df["Credit_History_Age_Months"] = convert_duration_to_months(
        df["Credit_History_Age"]
    )
    df = df.drop("Credit_History_Age", axis=1)
    print("✅ 'Credit_History_Age' column successfully converted and dropped")

    print("\n🏷️ CLEANING CATEGORICAL COLUMNS:")
    print("-" * 40)

    # Clean categorical columns
    categorical_columns = [
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
    ]
    for col in categorical_columns:
        print(f"Cleaning missing values in column '{col}'")
        df[col] = clean_missing_values(df[col])

    print("\n🔄 PROCESSING TYPE_OF_LOAN:")
    print("-" * 40)

    # Process Type_of_Loan column
    print("Processing 'Type_of_Loan' column to multi-hot encoding")
    df = process_type_of_loan(df)

    print("\n💾 SAVING RESULTS:")
    print("-" * 40)

    final_shape = df.shape
    print(f"Final data shape: {final_shape}")
    print(f"Shape change: {original_shape} → {final_shape}")

    print("\n" + "=" * 60)
    print("DATA CLEANING PROCESS COMPLETED")
    print("=" * 60)

    return df


def preprocess_data(
    raw_path: str,
    joblib_pipeline_save_path: str,
    target_col: str,
    cleaned_data_csv_save_path: str,
):
    DataCleaner = FunctionTransformer(clean_data)

    df = pd.read_csv(raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_col, axis=1),
        df[target_col],
        test_size=0.2,
        random_state=42,
        stratify=df[target_col],
    )


if __name__ == "__main__":
    df = pd.read_csv("Credit Score Dataset.csv")
    df = clean_data(df)
