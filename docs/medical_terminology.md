# Medical Superbill Data Extraction - Documentation

## Medical Billing Terminology and Essential Fields

### What is a Medical Superbill?

A superbill (also known as "Charge Slips," "Fee Tickets," or "Encounter Forms") is a detailed document that itemizes all healthcare services provided to a patient during a medical encounter. It serves as both a receipt for patients and documentation for insurance reimbursement claims.

### Essential Fields in Medical Superbills

#### 1. Patient Information
- **Patient Name** (First, Middle, Last)
- **Date of Birth (DOB)** - Critical for patient identification
- **Patient Address** (Street, City, State, ZIP Code)
- **Phone Number**
- **Insurance Information** (Insurance Company, Policy Number, Group Number)
- **Patient ID/Account Number**
- **Emergency Contact Information**

#### 2. Provider Information
- **Provider Name** (First, Last)
- **Provider NPI Number** (National Provider Identifier - 10-digit unique identifier)
- **Practice/Facility Name**
- **Provider Address**
- **Provider Phone Number**
- **Provider Email Address**
- **Tax ID Number**
- **Referring Provider Name and NPI** (if applicable)
- **Provider Signature**

#### 3. Visit/Service Information
- **Date of Service** - When treatment was provided
- **Place of Service (POS) Code** - Where treatment occurred
- **Time of Service** (Start/End times)
- **Visit Type** (New patient, Established patient, Follow-up, etc.)

#### 4. Medical Codes and Procedures
- **CPT Codes** (Current Procedural Terminology)
  - 5-digit numeric codes describing medical procedures
  - Maintained by American Medical Association (AMA)
  - Categories: Evaluation & Management, Anesthesia, Surgery, Radiology, Pathology, Medicine
- **ICD-10 Diagnosis Codes** (International Classification of Diseases, 10th Revision)
  - Format: Letter followed by 2 digits, optional decimal and 1-3 additional digits
  - Example: M54.5 (Low back pain)
- **HCPCS Codes** (Healthcare Common Procedure Coding System)
- **Modifier Codes** - Specify circumstances that alter procedures

#### 5. Financial Information
- **Charges/Fees** for each service
- **Units** or time spent for each procedure
- **Total Charges**
- **Amount Paid** (if any)
- **Outstanding Balance**
- **Payment Method**
- **Copayment Amount**
- **Deductible Information**

#### 6. Clinical Information
- **Chief Complaint**
- **Diagnosis Description**
- **Treatment Plan**
- **Medications Prescribed**
- **Follow-up Instructions**
- **Referrals Made**

### Protected Health Information (PHI) in Superbills

PHI includes any individually identifiable health information that is:
- Transmitted or maintained in any form (electronic, paper, oral)
- Created or received by healthcare providers, health plans, or healthcare clearinghouses

#### PHI Elements to Identify and Handle:
1. **Direct Identifiers:**
   - Patient names
   - Social Security Numbers (SSN)
   - Medical record numbers
   - Account numbers
   - Device identifiers and serial numbers

2. **Quasi-identifiers:**
   - Dates (birth, admission, discharge, death)
   - ZIP codes (especially if population < 20,000)
   - Ages over 89
   - Photographs and images

3. **Contact Information:**
   - Addresses (street, city, county, state)
   - Phone and fax numbers
   - Email addresses
   - Web URLs
   - IP addresses

### CPT Code Categories and Patterns

#### CPT Code Structure:
- **Format:** 5-digit numeric codes (00100-99999)
- **Categories:**
  - **Category I:** 00100-99499 (Most commonly used)
  - **Category II:** 0001F-9999F (Performance measurement)
  - **Category III:** 0001T-9999T (Temporary codes for emerging technology)

#### Common CPT Code Ranges:
- **99201-99499:** Evaluation and Management
- **00100-01999:** Anesthesia
- **10021-69990:** Surgery
- **70010-79999:** Radiology
- **80047-89398:** Pathology and Laboratory
- **90281-99607:** Medicine

### ICD-10 Diagnosis Code Structure

#### ICD-10-CM Format:
- **Structure:** [Letter][2 digits][optional decimal][1-3 additional digits]
- **Example:** M54.5 (Low back pain)
- **Chapters:** A00-Z99 organized by body system/condition type

#### Common ICD-10 Chapter Ranges:
- **A00-B99:** Infectious and parasitic diseases
- **C00-D49:** Neoplasms
- **E00-E89:** Endocrine, nutritional and metabolic diseases
- **F01-F99:** Mental, behavioral and neurodevelopmental disorders
- **G00-G99:** Diseases of the nervous system
- **I00-I99:** Diseases of the circulatory system
- **J00-J99:** Diseases of the respiratory system
- **K00-K95:** Diseases of the digestive system
- **M00-M99:** Diseases of the musculoskeletal system
- **S00-T88:** Injury, poisoning and certain other consequences

### HIPAA Compliance Requirements

#### Security Safeguards:
1. **Administrative Safeguards:**
   - Security officer designation
   - Workforce training
   - Access management
   - Audit controls

2. **Physical Safeguards:**
   - Facility access controls
   - Workstation security
   - Device and media controls

3. **Technical Safeguards:**
   - Access control
   - Audit controls
   - Integrity controls
   - Transmission security

#### Data Handling Requirements:
- **Minimum Necessary:** Only access/use PHI necessary for the task
- **Authorization:** Patient consent for uses beyond treatment, payment, operations
- **Accounting:** Track disclosures of PHI
- **Individual Rights:** Patient access to their own records

### Multi-Patient Document Characteristics

#### Common Scenarios:
1. **Family Practice Visits:** Multiple family members on single superbill
2. **Group Therapy Sessions:** Multiple patients in same session
3. **Batch Processing:** Multiple patients processed together
4. **Emergency Department:** Multiple patients on shift summary

#### Identification Patterns:
- **Patient separators:** Lines, boxes, or distinct sections
- **Repeated field patterns:** Name, DOB, Account# patterns
- **Sequential numbering:** Patient 1, Patient 2, etc.
- **Different handwriting styles:** Multiple providers
- **Varying appointment times:** Different service dates/times

### Quality Assurance Metrics

#### Accuracy Measurements:
- **Field Extraction Accuracy:** Percentage of correctly extracted fields
- **Code Validation Rate:** Percentage of valid CPT/ICD-10 codes
- **Patient Separation Accuracy:** Correct identification of patient boundaries
- **PHI Detection Rate:** Percentage of PHI correctly identified
- **False Positive Rate:** Incorrect extractions
- **Confidence Scoring:** Model certainty for each extraction

#### Performance Benchmarks:
- **Target Accuracy:** >95% for critical fields (Patient Name, CPT codes, ICD-10 codes)
- **Processing Speed:** <30 seconds per page
- **PHI Detection:** >99% sensitivity for common PHI patterns
- **Multi-patient Detection:** >90% accuracy in patient boundary identification
