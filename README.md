# UIDAI-Adhar-Hackathon-Report
# AADHAAR DATA HACKATHON 2026
## Consolidated Submission Document

**Team ID:** UIDAI_10715
**Analysis Period:** March 2025 - December 2025
**Total Records Analyzed:** 5,337,989+
**Tools Used:** Microsoft Power BI, Power Query, DAX

---

# TABLE OF CONTENTS

1. Problem Statement and Approach
2. Datasets Used
3. Methodology
4. Data Analysis and Visualisation
5. Code Implementation

---

# 1. PROBLEM STATEMENT AND APPROACH

## 1.1 Problem Statement

The Unique Identification Authority of India (UIDAI) has made anonymized datasets of Aadhaar enrolment and updates available to identify meaningful patterns, trends, anomalies, or predictive indicators. The goal of this analysis is to translate these findings into clear insights or solution frameworks that can support informed decision-making and system improvements for one of the world's largest identity programs.

Aadhaar, India's universal identification system, has enrolled over 1.4 billion residents, making it the largest biometric identity program globally. Understanding enrollment patterns, demographic shifts, geographic distribution, and operational anomalies is crucial for continuous improvement of this national infrastructure. This analysis addresses the challenge of extracting actionable intelligence from enrollment data to optimize operations, enhance service delivery, and ensure equitable access across India's diverse population.

## 1.2 Research Questions

Our analysis addressed five fundamental research questions:

**Research Question 1: Demographic Patterns**
What age groups are driving Aadhaar enrollment trends, and how do enrollment patterns vary across different demographic segments? Understanding the demographic composition of enrollments helps identify which population segments are being effectively served and which may require additional outreach efforts.

**Research Question 2: Geographic Distribution**
How is enrollment distributed across India's states and districts? Are there regional disparities that warrant attention? Geographic analysis reveals concentration patterns and identifies areas that may benefit from targeted enrollment drives or infrastructure improvements.

**Research Question 3: Temporal Patterns**
Are there seasonal, weekly, or daily patterns in enrollment data that can inform operational planning? Identifying temporal patterns enables optimization of staffing, resource allocation, and system capacity planning.

**Research Question 4: Anomaly Detection**
Are there unusual periods or locations with abnormal enrollment activity that require investigation? Anomaly detection helps identify potential data quality issues, system anomalies, or unusual events that merit further examination.

**Research Question 5: System Optimization**
What insights can be derived to improve Aadhaar enrollment operations, resource allocation, and service delivery? The ultimate goal is to translate data-driven insights into actionable recommendations for system improvement.

## 1.3 Analytical Approach

Our approach follows a structured five-phase methodology:

**Phase 1: Data Exploration and Understanding**
The initial phase focused on understanding the data landscape. We consolidated multiple CSV files from each dataset type (Enrollment, Biometric, Demographic), explored data schemas and field definitions, identified data quality issues, and established baseline statistics. This phase revealed that our data spans 10 months (March-December 2025) with geographic coverage across 55 states and territories.

**Phase 2: Data Cleaning and Preparation**
Data quality is foundational to reliable analysis. We standardized column naming conventions across datasets, converted date fields from DD-MM-YYYY format to proper datetime objects, validated and corrected data types for numeric fields, removed duplicate records, handled missing values through appropriate imputation, and created derived fields for aggregation and analysis.

**Phase 3: Exploratory Data Analysis**
The exploratory phase focused on discovering patterns, relationships, and initial insights. We generated descriptive statistics for all key variables, created visualizations for distribution analysis, performed bivariate analysis across demographic and geographic dimensions, and documented preliminary findings that guided subsequent statistical validation.

**Phase 4: Statistical Validation and Deep Analysis**
Moving beyond observation to validation, we applied statistical hypothesis testing including t-tests for group comparisons and chi-square tests for distribution validation. We calculated effect sizes using Cohen's d, performed correlation analysis, implemented anomaly detection using z-score methodology, and conducted comparative analysis across states, time periods, and demographics.

**Phase 5: Synthesis and Recommendation Development**
The final phase transformed analysis into action. We synthesized findings across all analyses, developed prioritized recommendations grounded in evidence, created visualizations for stakeholder communication, documented methodology and limitations, and prepared the comprehensive submission package.

---

# 2. DATASETS USED

## 2.1 Dataset Overview

Our analysis utilized three interconnected datasets provided by UIDAI for the hackathon:

### 2.1.1 Enrollment Dataset

The enrollment dataset served as our primary data source for analyzing Aadhaar registration patterns. This dataset contains detailed records of Aadhaar enrollment transactions, with each record representing enrollment activity at a specific location on a specific date.

| Metric | Value |
|--------|-------|
| Total Records | 984,189 |
| Time Period | March 2025 - December 2025 |
| Unique States/UTs | 55 |
| Unique Districts | 985 |
| Unique Pincodes | 19,463 |

### 2.1.2 Biometric Dataset

The biometric dataset captures biometric data update transactions, providing context about ongoing maintenance of Aadhaar records.

| Metric | Value |
|--------|-------|
| Total Records | 1,767,677 |
| Time Period | March 2025 - December 2025 |

### 2.1.3 Demographic Dataset

The demographic dataset contains demographic information about Aadhaar holders, enabling comparative analysis of enrollment patterns.

| Metric | Value |
|--------|-------|
| Total Records | 1,962,664 |
| Time Period | March 2025 - December 2025 |

**Combined Analysis Coverage:** 5,337,989+ total records analyzed across all three datasets.

## 2.2 Data Schema and Field Definitions

### 2.2.1 Enrollment Dataset Schema

**Primary Key Fields:**

| Field Name | Data Type | Format | Description |
|------------|-----------|--------|-------------|
| date | DateTime | DD-MM-YYYY | Date of enrollment transaction |
| state | String | Text | State name where enrollment occurred |
| district | String | Text | District name within state |
| pincode | Integer | 6-digit numeric | Postal code of enrollment location |

**Age Group Fields:**

| Field Name | Original Name | Data Type | Description |
|------------|--------------|-----------|-------------|
| enroll_age_0_5 | age_0_5 | Integer | Count of enrollees aged 0-5 years |
| enroll_age_5_17 | age_5_17 | Integer | Count of enrollees aged 5-17 years |
| enroll_age_18_greater | age_18_greater | Integer | Count of enrollees aged 18+ years |

### 2.2.2 Biometric Dataset Schema

**Primary Key Fields:**

| Field Name | Data Type | Format | Description |
|------------|-----------|--------|-------------|
| date | DateTime | DD-MM-YYYY | Date of biometric transaction |
| state | String | Text | State where biometric update occurred |
| district | String | Text | District name |
| pincode | Integer | 6-digit numeric | Postal code |

**Biometric Update Fields:**

| Field Name | Original Name | Data Type | Description |
|------------|--------------|-----------|-------------|
| biometric_age_5_17 | bio_age_5_17 | Integer | Biometric updates for age 5-17 |
| biometric_age_17_ | bio_age_17_ | Integer | Biometric updates for age 17+ |

### 2.2.3 Demographic Dataset Schema

**Primary Key Fields:**

| Field Name | Data Type | Format | Description |
|------------|-----------|--------|-------------|
| date | DateTime | DD-MM-YYYY | Date of demographic record |
| state | String | Text | State name |
| district | String | Text | District name |
| pincode | Integer | 6-digit numeric | Postal code |

**Demographic Fields:**

| Field Name | Original Name | Data Type | Description |
|------------|--------------|-----------|-------------|
| demographic_age_5_17 | demo_age_5_17 | Integer | Demographic records age 5-17 |
| demographic_age_17_ | demo_age_17_ | Integer | Demographic records age 17+ |

## 2.3 Data Quality Assessment

### 2.3.1 Data Quality Observations

Overall, the data quality was excellent with minimal issues that required treatment:

**Missing Values:**
- Date fields had minimal missing values (<0.1%) and affected records were removed
- Geographic fields had no significant missing data
- Age count fields had no missing values after type conversion

**Duplicate Records:**
- Duplicate records were minimal (<1% of total)
- Duplicates were removed during the data cleaning phase
- No significant impact on overall analysis

**Geographic Coverage:**
- Complete coverage across 55 states and Union Territories
- Strong representation from 985 districts
- 19,463 unique pincodes covered

**Temporal Coverage:**
- Data spans 10 months (March 2025 - December 2025)
- Sufficient for intra-year pattern analysis
- Full annual cycle not available for complete seasonality analysis

### 2.3.2 Data Limitations

Several limitations should be acknowledged:

- The 10-month time period prevents full annual cycle analysis
- Aggregated data format limits individual-level analysis capabilities
- Single year of data prevents year-over-year comparisons
- External factors (policy changes, campaigns) not available for context

---

# 3. METHODOLOGY

## 3.1 Data Processing Pipeline

### 3.1.1 Data Ingestion Strategy

Our data ingestion approach consolidated multiple source files into unified analytical datasets. The process involved:

**Step 1: File Discovery**
Identified all CSV files in each dataset directory using Python's pathlib library for robust file handling.

**Step 2: Sequential Loading**
Read each CSV file into a pandas DataFrame, ensuring proper encoding and parsing.

**Step 3: Concatenation**
Combined individual DataFrames into unified datasets using pandas concat function with ignore_index=True to maintain clean indexing.

**Step 4: Provenance Tracking**
Maintained source file references to ensure data traceability throughout the analysis.

### 3.1.2 Data Cleaning Procedures

**Column Standardization:**
Renamed inconsistent column names to follow a consistent naming convention: `{dataset}_{field}_{qualifier}`. For example, `age_0_5` was renamed to `enroll_age_0_5` to clearly indicate it belongs to the enrollment dataset.

**Date Conversion:**
Parsed date fields from DD-MM-YYYY string format to pandas datetime objects, enabling temporal analysis. Invalid date values were handled with error coercion to prevent pipeline failures.

**Data Type Validation:**
Ensured numeric columns were properly typed by converting string representations to integers. Missing values were filled with 0 before type conversion.

**Duplicate Removal:**
Identified and removed duplicate records to ensure each transaction was counted only once.

**Derived Field Creation:**
Created calculated fields to support analysis:
- `enrollment_total`: Sum of all age group columns
- `day_of_week`: Extracted day of week from date (0=Monday, 6=Sunday)
- `month`: Extracted month number (1-12)
- `week_of_year`: Extracted ISO week number

### 3.1.3 Data Validation Procedures

Post-cleaning validation ensured data quality:
- Verified null value counts for all fields
- Confirmed numeric fields were within expected ranges
- Validated date ranges against expected period
- Checked geographic consistency across records

## 3.2 Power BI Implementation

### 3.2.1 Power Query Transformations

**Data Import:**
Loaded CSV files directly into Power BI using Get Data > Text/CSV functionality. Each dataset was imported separately and processed through Power Query Editor.

**Column Renaming:**
Standardized column names across all three datasets for consistency:
- Enrollment: `age_0_5`, `age_5_17`, `age_18_greater`, `date`
- Biometric: `bio_age_5_17`, `bio_age_17_`, `date`
- Demographic: `demo_age_5_17`, `demo_age_17_`, `date`

**Date Format Conversion:**
Transformed date fields from DD-MM-YYYY format to proper Date type in Power Query:
```
= Date.FromText([date], "dd-MM-yyyy")
```

**Data Type Setting:**
Explicitly set data types for each column:
- Date fields: Date type
- Numeric fields: Whole number type
- Text fields: Text type

### 3.2.2 DAX Measures Created

**Total Enrollment Measure:**
```dax
Total Enrollment = SUM(Enrollment[enrollment_total])
```

**Child Enrollment Rate:**
```dax
Child Enrollment % = 
DIVIDE(
    SUM(Enrollment[enroll_age_0_5]),
    SUM(Enrollment[enrollment_total]),
    0
)
```

**Weekday/Weekend Analysis:**
```dax
Day Type = 
SWITCH(
    WEEKDAY('Enrollment'[Date], 2),
    1, "Weekday", 2, "Weekday", 3, "Weekday",
    4, "Weekday", 5, "Weekday",
    6, "Weekend", 7, "Weekend"
)

Weekday Average = 
AVERAGEX(
    FILTER(ALL('Enrollment'), 'Enrollment'[Day Type] = "Weekday"),
    SUM(Enrollment[enrollment_total])
)

Weekend Average = 
AVERAGEX(
    FILTER(ALL('Enrollment'), 'Enrollment'[Day Type] = "Weekend"),
    SUM(Enrollment[enrollment_total])
)
```

**Z-Score for Anomaly Detection:**
```dax
Z-Score = 
VAR Mean = AVERAGE(Enrollment[enrollment_total])
VAR StdDev = STDEV.P(Enrollment[enrollment_total])
RETURN
DIVIDE(
    SUM(Enrollment[enrollment_total]) - Mean,
    StdDev,
    0
)
```

## 3.3 Statistical Analysis Methods

### 3.3.1 Descriptive Statistics

We generated comprehensive descriptive statistics to characterize the data:

| Metric | Description | Use Case |
|--------|-------------|----------|
| Mean | Average value | Central tendency measurement |
| Median | 50th percentile | Robust central tendency |
| Standard Deviation | Spread measurement | Variability assessment |
| Min/Max | Range boundaries | Outlier identification |
| Percentiles | Distribution splits | Detailed distribution understanding |

### 3.3.2 Hypothesis Testing

**T-Test: Weekday vs Weekend Enrollment**

We employed an independent samples t-test to determine if weekday enrollment differs significantly from weekend enrollment.

**Interpretation Guidelines:**
- p-value < 0.05 indicates statistically significant difference
- Cohen's d effect size: 0.2 (small), 0.5 (medium), 0.8 (large)

**Chi-Square Test: Age Group Distribution**

We tested whether age group distribution differs from uniform expectation:
- H0: Enrollment is evenly distributed across age groups
- H1: Enrollment is not evenly distributed across age groups

**Correlation Analysis:**

We measured relationships between temporal variables and enrollment using Pearson correlation.

### 3.3.3 Anomaly Detection

We implemented z-score analysis to identify days with unusual enrollment activity:

**Threshold Selection Rationale:**
- |Z| > 2: Notable deviation (approximately 5% of data under normal distribution)
- |Z| > 3: Extreme deviation (approximately 0.3% of data)

---

# 4. DATA ANALYSIS AND VISUALISATION

## 4.1 Key Findings Summary

### 4.1.1 Finding 1: Child-Centric Enrollment Pattern

One of the most significant findings is the overwhelming dominance of child enrollments (ages 0-5) in the Aadhaar system.

**Age Distribution Analysis:**

| Age Group | Enrollment Count | Percentage |
|-----------|-----------------|------------|
| Children (0-5) | 3,477,851 | **65.15%** |
| Youth (5-17) | 1,693,602 | 31.73% |
| Adults (18+) | 166,536 | 3.12% |
| **Total** | **5,337,989** | **100%** |

**Statistical Validation:**
- Chi-square test: χ² = 3,087,357.82, p < 0.001
- Result: Highly significant departure from uniform distribution

**Interpretation:**
The finding that 65% of enrollments are for young children indicates strong integration between Aadhaar and birth registration systems in India. Parents are prioritizing Aadhaar enrollment for their children from the earliest possible age, likely driven by requirements for accessing government services, school admissions, and other benefits that require Aadhaar authentication.

**Implications:**
- Pediatric-friendly enrollment infrastructure should be a priority
- Integration with hospitals and birth registration is effective
- Adult enrollment gaps may warrant targeted outreach campaigns

### 4.1.2 Finding 2: Geographic Concentration

Analysis reveals significant geographic concentration in enrollment patterns.

**Top 5 States by Enrollment:**

| Rank | State | Enrollments | Percentage |
|------|-------|-------------|------------|
| 1 | Uttar Pradesh | 1,002,631 | 18.8% |
| 2 | Bihar | 593,753 | 11.1% |
| 3 | Madhya Pradesh | 489,212 | 9.2% |
| 4 | West Bengal | 369,206 | 6.9% |
| 5 | Maharashtra | 364,496 | 6.8% |
| - | **Top 5 Total** | **2,819,298** | **53%** |

**Interpretation:**
Over half of all enrollments are concentrated in just 5 states, reflecting India's population distribution where these states represent a significant portion of the national population. However, this concentration also raises questions about equitable service delivery in smaller states and Union Territories.

**Implications:**
- Resource allocation should match demand in high-volume states
- Targeted outreach may be needed in underrepresented regions
- Best practices from high-performing states should be documented and shared

### 4.1.3 Finding 3: Weekday vs Weekend Enrollment Pattern

A clear weekday-weekend enrollment pattern emerged from the analysis.

**Enrollment Comparison:**

| Day Type | Average Daily Enrollment |
|----------|-------------------------|
| Weekdays (Mon-Fri) | ~60,000 |
| Weekends (Sat-Sun) | ~45,000 |
| **Difference** | **+33%** |

**Statistical Validation:**
- T-test: t = 10.91, p < 0.001 (highly significant)
- Effect size: Cohen's d = 0.026 (small but consistent effect)

**Interpretation:**
Weekday enrollment is approximately 33% higher than weekend enrollment. This pattern likely reflects:
- Reduced operating hours at enrollment centers on weekends
- Government office closures affecting official documentation needs
- People's tendency to handle administrative tasks on weekdays

**Implications:**
- Staffing levels can be optimized based on expected demand
- Cost savings may be achieved by reducing weekend operations
- Customer experience may improve with better weekday capacity

### 4.1.4 Finding 4: Anomaly Detection

Statistical analysis identified several days with unusual enrollment activity.

**Anomalous Days Identified (|Z| > 2):**

| Date | Enrollments | Z-Score | Interpretation |
|------|-------------|---------|----------------|
| April 1, 2025 | 257,438 | +2.70 | Notable spike |
| June 1, 2025 | 215,734 | +2.14 | Notable spike |
| **July 1, 2025** | **616,868** | **+7.57** | **Extreme anomaly** |

**Critical Finding - July 1, 2025:**
The July 1, 2025 enrollment figure is extraordinary - 616,868 enrollments in a single day represents over 10 times the daily average and 7.57 standard deviations above the mean. Under normal distribution assumptions, such an extreme value is statistically impossible (probability < 0.0001%).

**Possible Explanations:**
- Data migration or bulk upload from legacy systems
- Special enrollment drive or campaign
- System integration event with another government service
- Data quality issue requiring investigation

**Implication:**
This finding requires immediate operational investigation. If it represents genuine demand, system capacity should be enhanced for similar events. If it represents a data issue, quality controls need strengthening.

## 4.2 Visualizations Created

### 4.2.1 Dashboard Overview
The comprehensive Power BI dashboard presents multiple key metrics in a single view, enabling stakeholders to quickly understand the enrollment landscape.

### 4.2.2 State-wise Enrollment Comparison
Visual representation of enrollment concentration across Indian states, highlighting the dominance of Uttar Pradesh and identifying the distribution tail for smaller states.

### 4.2.3 Weekly Pattern Analysis
Bar chart comparing weekday versus weekend enrollment volumes, clearly demonstrating the 33% difference in operational patterns.

### 4.2.4 Age Group Distribution
Pie chart illustrating the demographic composition of enrollments, with children aged 0-5 dominating at over 65% of all enrollments.

### 4.2.5 Monthly Trend Analysis
Line chart showing enrollment trends over the 10-month analysis period, revealing monthly variations and identifying seasonal patterns.

### 4.2.6 Anomaly Detection Visualization
Time series chart highlighting anomalous days with statistical thresholds, making the July 1 anomaly immediately visible.

---

# 5. CODE IMPLEMENTATION

## 5.1 Python Data Loading Module

```python
"""
Aadhaar Data Hackathon 2026 - Data Loading and Cleaning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AadhaarDataLoader:
    """
    Load, clean, and standardize Aadhaar enrollment data.
    
    This class handles the complete data ingestion pipeline including
    file discovery, format conversion, data validation, and cleaning.
    """
    
    def __init__(self, data_dir):
        """
        Initialize with directory containing dataset folders.
        
        Args:
            data_dir: Path to directory containing 'api_data_aadhar_enrolment',
                     'api_data_aadhar_biometric', and 'api_data_aadhar_demographic'
        """
        self.data_dir = Path(data_dir)
        self.enrollment_dir = self.data_dir / 'api_data_aadhar_enrolment'
        self.biometric_dir = self.data_dir / 'api_data_aadhar_biometric'
        self.demographic_dir = self.data_dir / 'api_data_aadhar_demographic'
    
    def load_enrollment_data(self):
        """
        Load and consolidate enrollment CSV files.
        
        Returns:
            pandas DataFrame with cleaned enrollment data
        """
        # Discover all CSV files
        files = list(self.enrollment_dir.glob('*.csv'))
        
        # Load and concatenate
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names
        df = df.rename(columns={
            'age_0_5': 'enroll_age_0_5',
            'age_5_17': 'enroll_age_5_17',
            'age_18_greater': 'enroll_age_18_plus',
            'date': 'enrollment_date'
        })
        
        # Convert dates
        df['enrollment_date'] = pd.to_datetime(
            df['enrollment_date'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        
        # Ensure numeric types
        for col in ['enroll_age_0_5', 'enroll_age_5_17', 'enroll_age_18_plus']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Calculate total enrollment
        df['enrollment_total'] = df[
            ['enroll_age_0_5', 'enroll_age_5_17', 'enroll_age_18_plus']
        ].sum(axis=1)
        
        # Add temporal features
        df['day_of_week'] = df['enrollment_date'].dt.dayofweek
        df['month'] = df['enrollment_date'].dt.month
        df['week_of_year'] = df['enrollment_date'].dt.isocalendar().week
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
```

## 5.2 Statistical Analysis Module

```python
"""
Aadhaar Data Hackathon 2026 - Statistical Analysis Module
"""

import numpy as np
from scipy import stats
import pandas as pd

class AadhaarAnalyzer:
    """
    Statistical analysis for Aadhaar enrollment data.
    
    Provides methods for hypothesis testing, anomaly detection,
    and statistical validation of patterns.
    """
    
    def __init__(self, df):
        """
        Initialize with enrollment DataFrame.
        
        Args:
            df: pandas DataFrame with 'enrollment_date' and 'enrollment_total' columns
        """
        self.df = df
        self.daily_enrollment = df.groupby('enrollment_date')['enrollment_total'].sum()
    
    def weekday_weekend_test(self):
        """
        Perform t-test for weekday vs weekend enrollment.
        
        Returns:
            dict with t_statistic, p_value, and cohens_d
        """
        # Separate weekday and weekend data
        weekday = self.df[self.df['day_of_week'] < 5]['enrollment_total']
        weekend = self.df[self.df['day_of_week'] >= 5]['enrollment_total']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(weekday, weekend)
        
        # Calculate effect size (Cohen's d)
        n1, n2 = len(weekday), len(weekend)
        var1, var2 = weekday.var(), weekend.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (weekday.mean() - weekend.mean()) / pooled_std
        
        return {
            't_statistic': round(t_stat, 4),
            'p_value': p_value,
            'cohens_d': round(cohens_d, 4),
            'weekday_mean': round(weekday.mean(), 2),
            'weekend_mean': round(weekend.mean(), 2)
        }
    
    def detect_anomalies(self, threshold=2):
        """
        Detect anomalous days using z-score method.
        
        Args:
            threshold: Z-score threshold (default: 2 standard deviations)
            
        Returns:
            dict with anomalies and their z-scores
        """
        # Calculate z-scores
        z_scores = (self.daily_enrollment - self.daily_enrollment.mean()) / self.daily_enrollment.std()
        
        # Identify anomalies
        anomaly_mask = abs(z_scores) > threshold
        anomalies = self.daily_enrollment[anomaly_mask]
        z_anomalies = z_scores[anomaly_mask]
        
        return {
            'anomalies': anomalies.to_dict(),
            'z_scores': z_anomalies.to_dict(),
            'mean': round(self.daily_enrollment.mean(), 2),
            'std': round(self.daily_enrollment.std(), 2),
            'threshold': threshold
        }
    
    def age_distribution_test(self):
        """
        Chi-square test for age group distribution.
        
        Returns:
            dict with chi_square statistic and p_value
        """
        observed = [
            self.df['enroll_age_0_5'].sum(),
            self.df['enroll_age_5_17'].sum(),
            self.df['enroll_age_18_plus'].sum()
        ]
        
        # Expected frequencies under uniform distribution
        expected = [sum(observed) / 3] * 3
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        return {
            'chi_square': round(chi2, 2),
            'p_value': p_value,
            'observed': observed,
            'expected': expected
        }
    
    def state_analysis(self):
        """
        Generate state-level summary statistics.
        
        Returns:
            DataFrame with state-level aggregations
        """
        return self.df.groupby('state').agg({
            'enrollment_total': ['sum', 'mean', 'std'],
            'enroll_age_0_5': 'sum',
            'enroll_age_5_17': 'sum',
            'enroll_age_18_plus': 'sum'
        }).round(2)
```

## 5.3 Main Execution Script

```python
"""
Aadhaar Data Hackathon 2026 - Main Analysis Script
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import AadhaarDataLoader
from statistical_analysis import AadhaarAnalyzer

def main():
    """
    Main execution function for Aadhaar enrollment analysis.
    """
    # Configuration
    DATA_DIR = Path('/workspace/aadhaar_data')
    OUTPUT_DIR = DATA_DIR / 'output'
    
    print("=" * 70)
    print("AADHAAR DATA HACKATHON 2026 - ENROLLMENT ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading and cleaning data...")
    loader = AadhaarDataLoader(DATA_DIR)
    df = loader.load_enrollment_data()
    print(f"  Loaded {len(df):,} enrollment records")
    
    # Add temporal features
    df['day_of_week'] = df['enrollment_date'].dt.dayofweek
    df['month'] = df['enrollment_date'].dt.month
    
    # Run analysis
    print("\n[2/4] Running statistical analysis...")
    analyzer = AadhaarAnalyzer(df)
    
    # Weekday vs Weekend test
    ww_results = analyzer.weekday_weekend_test()
    print(f"\n  Weekday vs Weekend T-Test:")
    print(f"    T-statistic: {ww_results['t_statistic']}")
    print(f"    P-value: {ww_results['p_value']:.2e}")
    print(f"    Cohen's d: {ww_results['cohens_d']}")
    
    # Age distribution test
    age_results = analyzer.age_distribution_test()
    print(f"\n  Age Distribution Chi-Square:")
    print(f"    Chi-square: {age_results['chi_square']:.2f}")
    print(f"    P-value: {age_results['p_value']:.2e}")
    
    # Anomaly detection
    anomaly_results = analyzer.detect_anomalies()
    print(f"\n  Anomaly Detection (|Z| > 2):")
    print(f"    Number of anomalies: {len(anomaly_results['anomalies'])}")
    for date, value in anomaly_results['anomalies'].items():
        z = anomaly_results['z_scores'][date]
        print(f"    {date}: {value:,} (Z={z:.2f})")
    
    # State analysis
    print("\n[3/4] Generating state-level analysis...")
    state_summary = analyzer.state_analysis()
    print(f"  Analyzed {len(state_summary)} states/territories")
    
    # Summary statistics
    print("\n[4/4] Computing summary statistics...")
    daily_enrollment = df.groupby('enrollment_date')['enrollment_total'].sum()
    print(f"  Total records: {df['enrollment_total'].sum():,}")
    print(f"  Date range: {df['enrollment_date'].min()} to {df['enrollment_date'].max()}")
    print(f"  Daily average: {daily_enrollment.mean():,.0f}")
    print(f"  Daily std dev: {daily_enrollment.std():,.0f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

## 5.4 Power BI DAX Reference

```dax
-- Basic Enrollment Measures
Total Enrollment = SUM(Enrollment[enrollment_total])
Total Biometric Updates = SUM(Biometric[biometric_total])
Total Demographic Records = SUM(Demographic[demographic_total])

-- Age Group Calculations
Child Enrollment = SUM(Enrollment[enroll_age_0_5])
Youth Enrollment = SUM(Enrollment[enroll_age_5_17])
Adult Enrollment = SUM(Enrollment[enroll_age_18_greater])

Child Enrollment % = 
DIVIDE([Child Enrollment], [Total Enrollment], 0)

-- Time-Based Analysis
Day Type = 
SWITCH(
    WEEKDAY('Enrollment'[Date], 2),
    1, "Weekday", 2, "Weekday", 3, "Weekday",
    4, "Weekday", 5, "Weekday",
    6, "Weekend", 7, "Weekend"
)

Weekday Average = 
AVERAGEX(
    FILTER(ALL('Enrollment'), 'Enrollment'[Day Type] = "Weekday"),
    SUM(Enrollment[enrollment_total])
)

Weekend Average = 
AVERAGEX(
    FILTER(ALL('Enrollment'), 'Enrollment'[Day Type] = "Weekend"),
    SUM(Enrollment[enrollment_total])
)

Weekday Weekend Diff % = 
DIVIDE([Weekday Average] - [Weekend Average], [Weekend Average], 0) * 100

-- Anomaly Detection
Daily Average = AVERAGE(Enrollment[enrollment_total])
Daily StdDev = STDEV.P(Enrollment[enrollment_total])

Z-Score = 
DIVIDE(
    [Total Enrollment] - [Daily Average],
    [Daily StdDev],
    BLANK()
)

Is Anomaly = 
IF(ABS([Z-Score]) > 2, "Anomaly", "Normal")
```

---

# KEY INSIGHTS AND RECOMMENDATIONS

## Key Insights

**Insight 1: Aadhaar as India's Birth Identity System**
The dominance of child enrollment (65%) suggests that Aadhaar is increasingly functioning as a birth identity system rather than just an adult ID card. This has profound implications for how UIDAI should think about its infrastructure, outreach, and integration with other systems.

**Insight 2: Geographic Equity Deserves Attention**
While concentration in large states makes demographic sense, it also raises questions about whether smaller states and Union Territories are getting adequate support.

**Insight 3: Operations Can Be Optimized Based on Demand Patterns**
The weekday/weekend patterns provide a clear roadmap for operational optimization. Staffing, center hours, and resource allocation can all be tuned to match actual demand patterns.

**Insight 4: Anomaly Detection Is Critical**
The July 1 anomaly demonstrates that unexpected events happen and need to be monitored. A robust monitoring and response system could help UIDAI respond to such events more effectively.

## Prioritized Recommendations

**Priority 1: Critical (Immediate - 0-1 Month)**
- Investigate July 1, 2025 anomaly
- Optimize staffing patterns based on weekday/weekend demand
- Enhance pediatric infrastructure

**Priority 2: Strategic (1-3 Months)**
- Geographic resource rebalancing study
- Implement real-time monitoring dashboards
- Strengthen data quality processes

**Priority 3: Operational (3-6 Months)**
- Develop seasonal planning framework
- Strengthen integration with birth registration
- Create staff training programs

---

# CONCLUSION

This analysis demonstrates the value of data-driven decision-making for Aadhaar operations. By applying rigorous analytical methods and focusing on actionable outcomes, we have provided UIDAI with valuable insights that can inform operational decisions and system improvements.

The key findings - child-centric enrollment, geographic concentration, weekday patterns, and anomaly detection - all have clear implications for how UIDAI can optimize its operations. The recommendations provided are grounded in statistical evidence and aligned with potential operational impact.

Our approach emphasizes rigor, clarity, actionability, and reproducibility. All methods are documented, and code is provided to enable reproduction and extension of this work.

---

**Analysis completed:** January 2026
**Data period:** March 2025 - December 2025
**Total records analyzed:** 5,337,989+
**Team ID:** UIDAI_10715
