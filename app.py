import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit import session_state as ss
from openai import AzureOpenAI
import time
import os
from dotenv import load_dotenv
from io import BytesIO
import json
import re
import string
from difflib import SequenceMatcher

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Doctor Review", page_icon="🏥", layout="wide")
st.title("Doctor Information Management with AI Analysis & Entity Matching")
st.markdown("---")

REVIEWER = "Albert Yao"

# Categories definition
CATEGORIES = {
    "Address Change": "Doctor changed their primary clinic address.",
    "Contact Information Update": "Doctor updated their phone number and/or email address.",
    "Specialty Update": "Doctor added or changed their medical specialty.",
    "Clinic Affiliation Change": "Doctor switched their primary clinic affiliation.",
    "License or NPI Update": "Doctor updated their NPI number and/or renewed their license.",
    "Status Change": "Doctor changed status (Active/Inactive/Retired).",
    "Verification Update": "Doctor's profile was verified via phone call or other method.",
    "Name Correction": "Doctor corrected their name (first, middle, or last).",
    "New Doctor Onboarding": "New doctor was added to the system.",
    "Clinic Address Correction": "Corrected a typo or error in clinic address.",
    "Clinic Name Correction": "Updated or corrected the clinic name.",
    "Date of Birth Correction": "Corrected the doctor's date of birth.",
    "Role or Title Update": "Updated the doctor's role or title.",
    "System-Initiated Update": "System auto-updated verification status.",
    "Multiple Field Update": "Multiple fields changed in one update."
}

# Initialize session state
if "excel_file" not in ss:
    ss.excel_file = None
if "excel_file_path" not in ss:
    ss.excel_file_path = None
if "df" not in ss:
    ss.df = None
if "analyzed_data" not in ss:
    ss.analyzed_data = None
if "audit" not in ss:
    ss.audit = []
if "selected_row_idx" not in ss:
    ss.selected_row_idx = None
if "match_results" not in ss:
    ss.match_results = None
if "changes_preview" not in ss:
    ss.changes_preview = None
if "azure_config" not in ss:
    # Load Azure OpenAI configuration from environment variables
    ss.azure_config = {
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "endpoint": os.getenv("OPENAI_DEPLOYMENT_ENDPOINT", "").strip(),
        "deployment_name": os.getenv("OPENAI_DEPLOYMENT_NAME", "").strip(),
        "api_version": os.getenv("OPENAI_DEPLOYMENT_VERSION", "").strip()
    }

# ==================== ENTITY MATCHING FUNCTIONS ====================
def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def digits_only(s):
    if pd.isna(s): return ""
    return re.sub(r"\D", "", str(s))

def get_zip5(z):
    z = digits_only(z)
    return z[:5] if z else ""

def similarity(a, b):
    return SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()

def token_overlap(a, b):
    def tokens(s):
        return set(re.findall(r"[a-z0-9]+", clean_text(s)))
    A, B = tokens(a), tokens(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def soundex(name):
    s = clean_text(name)
    if not s: return ""
    first = s[0].upper()
    mp = {**{k:"1" for k in "bfpv"}, **{k:"2" for k in "cgjkqsxz"},
          **{k:"3" for k in "dt"}, "l":"4", "m":"5", "n":"5", "r":"6"}
    digits, prev = [], mp.get(s[0], "")
    for ch in s[1:]:
        d = mp.get(ch, "")
        if d and d != prev: digits.append(d)
        prev = d
    return (first + "".join(digits) + "000")[:4]

WEIGHTS = {
    "npi_eq": 1.00, "email_eq": 0.70, "phone_eq": 0.60,
    "last_sim": 0.40, "first_sim": 0.25, "name_token": 0.20,
    "zip_eq": 0.25, "street_sim": 0.25, "city_eq": 0.05,
    "state_eq": 0.05, "spec_sim": 0.10, "school_sim": 0.05,
}

def match_record_in_df(df, first_name=None, last_name=None, email=None, phone=None, 
                       npi=None, specialty=None, city=None, state=None, zip_code=None, top_n=10):
    """Match a record against the loaded DataFrame."""

    # Detect specialty columns
    spec1_col, spec2_col = None, None
    for c in df.columns:
        if re.search(r'primary.*specialty', str(c), re.I): 
            spec1_col = c
        elif re.search(r'secondary.*specialty', str(c), re.I): 
            spec2_col = c
    if not spec1_col:
        for c in df.columns:
            if re.search(r'specialt(y|ies)|taxonomy|dept', str(c), re.I) and spec1_col is None:
                spec1_col = c

    # Prepare query
    q = {
        "first": str(first_name or "").strip(),
        "last": str(last_name or "").strip(),
        "email": str(email or "").lower().strip(),
        "phone": digits_only(phone or ""),
        "npi": str(npi or "").strip(),
        "specialty": str(specialty or "").lower().strip(),
        "city": str(city or "").strip(),
        "state": str(state or "").strip(),
        "zip5": get_zip5(zip_code or ""),
    }

    # Score each candidate
    results = []
    for idx, row in df.iterrows():
        features = {}
        score = 0

        # Name features
        if q["first"] and "First_Name" in df.columns:
            features["first_sim"] = similarity(q["first"], str(row.get("First_Name", "")))
            score += WEIGHTS["first_sim"] * features["first_sim"]
        if q["last"] and "Last_Name" in df.columns:
            features["last_sim"] = similarity(q["last"], str(row.get("Last_Name", "")))
            score += WEIGHTS["last_sim"] * features["last_sim"]

        # Specialty - check BOTH primary and secondary
        if q["specialty"]:
            spec_scores = []
            if spec1_col and row.get(spec1_col, ""):
                spec_scores.append(similarity(q["specialty"], str(row[spec1_col])))
            if spec2_col and row.get(spec2_col, ""):
                spec_scores.append(similarity(q["specialty"], str(row[spec2_col])))
            if spec_scores:
                features["spec_sim"] = max(spec_scores)
                score += WEIGHTS["spec_sim"] * features["spec_sim"]

        # Email
        if q["email"] and "Email_Address" in df.columns:
            if q["email"] == str(row.get("Email_Address", "")).lower().strip():
                features["email_eq"] = 1.0
                score += WEIGHTS["email_eq"]

        # Phone
        if q["phone"] and "Phone_Number" in df.columns:
            if q["phone"] == digits_only(row.get("Phone_Number", "")):
                features["phone_eq"] = 1.0
                score += WEIGHTS["phone_eq"]

        # NPI
        if q["npi"] and "NPI" in df.columns:
            if q["npi"] == str(row.get("NPI", "")).strip():
                features["npi_eq"] = 1.0
                score += WEIGHTS["npi_eq"]

        # City
        if q["city"] and "City" in df.columns:
            if clean_text(q["city"]) == clean_text(str(row.get("City", ""))):
                features["city_eq"] = 1.0
                score += WEIGHTS["city_eq"]

        # State
        if q["state"] and "State" in df.columns:
            if clean_text(q["state"]) == clean_text(str(row.get("State", ""))):
                features["state_eq"] = 1.0
                score += WEIGHTS["state_eq"]

        # ZIP
        if q["zip5"] and "Zip" in df.columns:
            if q["zip5"] == get_zip5(str(row.get("Zip", ""))):
                features["zip_eq"] = 1.0
                score += WEIGHTS["zip_eq"]

        # Classify
        if score >= 0.8:
            status = "match"
        elif score >= 0.6:
            status = "review"
        else:
            status = "nonmatch"

        results.append({
            "Row": idx,
            "Score": round(score, 4),
            "Status": status,
            "Name": f"{row.get('First_Name', '')} {row.get('Last_Name', '')}",
            "Primary_Specialty": row.get(spec1_col, "") if spec1_col else "",
            "Secondary_Specialty": row.get(spec2_col, "") if spec2_col else "",
            "Email": row.get("Email_Address", ""),
            "Phone": row.get("Phone_Number", ""),
            "City": row.get("City", ""),
            "State": row.get("State", ""),
        })

    results_df = pd.DataFrame(results).sort_values("Score", ascending=False).head(top_n)
    return results_df.reset_index(drop=True)

# ==================== OPENAI ANALYSIS FUNCTIONS ====================
def extract_details_from_text(text, excel_columns, azure_config):
    """Extract relevant details from user text based on Excel file metadata."""
    try:
        if not all(azure_config.values()):
            st.error("⚠️ Azure OpenAI configuration is incomplete. Please check your .env file.")
            return None

        client = AzureOpenAI(
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["endpoint"]
        )

        columns_list = "\n".join([f"- {col}" for col in excel_columns])

        prompt = f"""You are analyzing a doctor profile update request. Extract any relevant information that matches the fields in the Excel database.

AVAILABLE EXCEL COLUMNS:
{columns_list}

UPDATE REQUEST TEXT:
"{text}"

INSTRUCTIONS:
- Carefully read the update request text
- Identify any information that corresponds to the Excel columns listed above
- Extract the values mentioned in the text
- ONLY include fields where you find explicit information in the text
- If a field is not mentioned or cannot be determined, do NOT include it

Respond in JSON format with extracted field-value pairs:
{{
  "field_name": "extracted_value"
}}

Use the EXACT column names from the Excel columns list above.
If no relevant information is found, return an empty object: {{}}

Respond with ONLY the JSON object, no other text.

JSON:"""

        response = client.chat.completions.create(
            model=azure_config["deployment_name"],
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Extract information from text and map it to database fields. Only extract information that is explicitly stated. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response


        if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            extracted_data = json.loads(response_text)
            validated_data = {}
            for field, value in extracted_data.items():
                if field in excel_columns:
                    validated_data[field] = value
                else:
                    for col in excel_columns:
                        if col.lower() == field.lower():
                            validated_data[col] = value
                            break

            return validated_data

        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing extraction response: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error extracting details: {str(e)}")
        return None

def analyze_text_with_openai(text, azure_config):
    """Use Azure OpenAI to analyze the text and map it to a category with confidence score."""
    try:
        if not all(azure_config.values()):
            st.error("⚠️ Azure OpenAI configuration is incomplete. Please check your .env file.")
            return None, 0.0

        client = AzureOpenAI(
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["endpoint"]
        )

        categories_list = "\n".join([f"{i+1}. {cat}: {desc}" for i, (cat, desc) in enumerate(CATEGORIES.items())])

        prompt = f"""Analyze the following doctor profile update request and categorize it.

AVAILABLE CATEGORIES:
{categories_list}

UPDATE REQUEST TEXT:
"{text}"

INSTRUCTIONS:
- Read the update request carefully
- Match it to the most appropriate category from the list above
- Provide a confidence score (0.5-1.0)

Respond in JSON format:
{{"category": "Category Name", "confidence": 0.XX}}

JSON:"""

        response = client.chat.completions.create(
            model=azure_config["deployment_name"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes doctor profile updates. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response


        if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
            category = result.get("category", "").strip()
            confidence = float(result.get("confidence", 0.5))

            if category not in CATEGORIES:
                for cat in CATEGORIES.keys():
                    if cat.lower() in category.lower() or category.lower() in cat.lower():
                        category = cat
                        break
                else:
                    category = "Multiple Field Update"
                    confidence = 0.5

            return category, confidence

        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing AI response: {str(e)}")
            return "Multiple Field Update", 0.5

    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None, 0.0

# ==================== FUNCTION TO SAVE EXCEL FILE ====================
def save_excel_file(df, file_path):
    """Save DataFrame back to the original Excel file, overwriting it."""
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        return True
    except Exception as e:
        st.error(f"Error saving Excel file: {str(e)}")
        return False

# ==================== UI for File Upload ====================
st.sidebar.header("📁 Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'], help="Upload the Excel file containing doctor information")
if uploaded_file is not None:
    try:
        if ss.df is None:
            ss.df = pd.read_excel(uploaded_file)
            # Save file to temp location so we can overwrite it
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            ss.excel_file_path = temp_path
        ss.excel_file = uploaded_file.name
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        st.sidebar.write(f"📊 Rows: {len(ss.df)}, Columns: {len(ss.df.columns)}")

    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Available Categories")
with st.sidebar.expander("View all categories"):
    for i, (cat, desc) in enumerate(CATEGORIES.items(), 1):
        st.sidebar.write(f"**{i}. {cat}**")
        st.sidebar.caption(desc)
        st.sidebar.markdown("")

# ==================== Doctor Information Input Section ====================
st.header("👨‍⚕️ Doctor Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    first_name = st.text_input("First Name", placeholder="Enter first name", help="Doctor's first name")

with col2:
    last_name = st.text_input("Last Name", placeholder="Enter last name", help="Doctor's last name")

with col3:
    email_address = st.text_input("Email Address", placeholder="doctor@example.com", help="Doctor's email address")

with col4:
    specialty = st.text_input("Specialty", placeholder="Cardiology", help="Medical specialty")

st.markdown("---")

# ==================== ENTITY MATCHING SECTION ====================
st.header("🔍 Find Best Match from MDM Table")

if ss.df is not None and (first_name or last_name or email_address or specialty):
    if st.button("🔎 Search for Matches", type="primary", use_container_width=True):
        with st.spinner("Searching for matches..."):
            ss.match_results = match_record_in_df(
                ss.df,
                first_name=first_name,
                last_name=last_name,
                email=email_address,
                specialty=specialty,
                top_n=5
            )

    if ss.match_results is not None and not ss.match_results.empty:
        st.subheader("Top Suggested Matches")

        # Display results table
        display_cols = ["Row", "Score", "Status", "Name", "Primary_Specialty", "Secondary_Specialty", "Email", "City", "State"]
        available_cols = [col for col in display_cols if col in ss.match_results.columns]
        st.dataframe(ss.match_results[available_cols], use_container_width=True, hide_index=True)

        # Radio button to select match
        st.write("### Select the Matching Doctor")
        selected_idx = st.radio(
            "Choose the row number of the doctor you want to update:",
            ss.match_results["Row"].tolist(),
            format_func=lambda x: f"Row {x}: {ss.match_results[ss.match_results['Row']==x]['Name'].values[0]} (Score: {ss.match_results[ss.match_results['Row']==x]['Score'].values[0]})",
            help="Select the row that best matches the doctor you're looking for"
        )

        ss.selected_row_idx = selected_idx
        st.success(f"✅ Selected Row: {selected_idx}")

        # Show selected row details
        selected_row_data = ss.df.iloc[selected_idx]
        st.write("### Selected Doctor Details")
        st.dataframe(pd.DataFrame([selected_row_data]), use_container_width=True)

    elif ss.match_results is not None:
        st.warning("⚠️ No good matches found. Try different search criteria.")
else:
    if ss.df is None:
        st.info("ℹ️ Please upload an Excel file in the sidebar first")
    else:
        st.info("ℹ️ Enter at least one search field (name, email, or specialty) to find matches")

st.markdown("---")

# ==================== Text Analysis Section ====================
st.header("🤖 Analyze Update Request")

user_input = st.text_area(
    "Enter the update request text:",
    placeholder="e.g., Doctor Smith changed their primary clinic address from Hoboken to Jersey City.",
    height=100,
    help="Paste or type the doctor update request text here"
)

can_analyze = (
    user_input and 
    ss.df is not None and 
    ss.selected_row_idx is not None and
    all(ss.azure_config.values())
)

analyze_btn = st.button(
    "🔍 Analyze with AI", 
    type="primary", 
    disabled=not can_analyze,
    use_container_width=True
)

if ss.df is None:
    st.warning("⚠️ Please upload an Excel file in the sidebar")
elif ss.selected_row_idx is None:
    st.info("ℹ️ Please select a matching doctor from the search results above")
elif not user_input:
    st.info("💡 Please enter the update request text to analyze")
elif not all(ss.azure_config.values()):
    st.error("❌ Azure OpenAI configuration is missing. Please check your .env file.")

if analyze_btn:
    with st.spinner("🔄 Analyzing with Azure OpenAI..."):
        extracted_details = None
        if ss.df is not None:
            extracted_details = extract_details_from_text(user_input, ss.df.columns.tolist(), ss.azure_config)

            if extracted_details:
                st.info(f"📝 **Extracted {len(extracted_details)} field(s) from the text:**")
                for field, value in extracted_details.items():
                    st.write(f"  **{field}:** {value}")

        category, confidence = analyze_text_with_openai(user_input, ss.azure_config)

    if category and ss.selected_row_idx is not None:
        row_data = ss.df.iloc[ss.selected_row_idx].copy()

        # Build changes preview - Use dict to prevent duplicates
        changes_dict = {}
        
        # Add extracted details first
        if extracted_details:
            for field, new_value in extracted_details.items():
                if field in ss.df.columns:
                    old_value = row_data.get(field, "")
                    if str(old_value) != str(new_value):
                        changes_dict[field] = {
                            "Field": field,
                            "Before": str(old_value) if pd.notna(old_value) else "",
                            "After": str(new_value)
                        }

        # Add manual entries - only if NOT already in changes_dict
        if first_name and "First_Name" in ss.df.columns and "First_Name" not in changes_dict:
            old_val = row_data.get("First_Name", "")
            if str(old_val) != str(first_name):
                changes_dict["First_Name"] = {
                    "Field": "First_Name",
                    "Before": str(old_val) if pd.notna(old_val) else "",
                    "After": first_name
                }
        
        if last_name and "Last_Name" in ss.df.columns and "Last_Name" not in changes_dict:
            old_val = row_data.get("Last_Name", "")
            if str(old_val) != str(last_name):
                changes_dict["Last_Name"] = {
                    "Field": "Last_Name",
                    "Before": str(old_val) if pd.notna(old_val) else "",
                    "After": last_name
                }
        
        if email_address and "Email_Address" in ss.df.columns and "Email_Address" not in changes_dict:
            old_val = row_data.get("Email_Address", "")
            if str(old_val) != str(email_address):
                changes_dict["Email_Address"] = {
                    "Field": "Email_Address",
                    "Before": str(old_val) if pd.notna(old_val) else "",
                    "After": email_address
                }
        
        if specialty:
            # Find specialty column
            specialty_col_found = False
            for col in ss.df.columns:
                if re.search(r'primary.*specialty', col, re.I):
                    if col not in changes_dict:
                        old_val = row_data.get(col, "")
                        if str(old_val) != str(specialty):
                            changes_dict[col] = {
                                "Field": col,
                                "Before": str(old_val) if pd.notna(old_val) else "",
                                "After": specialty
                            }
                    specialty_col_found = True
                    break

        # Convert dict to list
        changes_preview = list(changes_dict.values())

        ss.analyzed_data = {
            "row_index": ss.selected_row_idx,
            "category": category,
            "confidence": confidence,
            "text": user_input,
            "first_name": first_name if first_name else "",
            "last_name": last_name if last_name else "",
            "email": email_address if email_address else "",
            "specialty": specialty if specialty else "",
            "extracted_details": extracted_details if extracted_details else {},
            "row_data": row_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ss.changes_preview = changes_preview

        st.success(f"✅ Categorized as: **{category}** (Confidence: {confidence:.1%})")
        st.info("💡 Proposed changes ready for review. Click 'Accept' to apply them to the Excel file.")
        time.sleep(0.5)
        st.rerun()
    else:
        st.error("❌ Failed to analyze text. Please check your Azure OpenAI configuration and try again.")

# ==================== Display Analyzed Row ====================
if ss.analyzed_data is not None:
    st.markdown("---")
    st.header("📋 Review Proposed Changes")

    data = ss.analyzed_data

    st.subheader("👤 Doctor Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**First Name:** {data['first_name']}")
    with col2:
        st.info(f"**Last Name:** {data['last_name']}")
    with col3:
        st.info(f"**Email:** {data['email']}")
    with col4:
        st.info(f"**Specialty:** {data.get('specialty', '')}")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("📂 Category", data["category"])
        st.metric("📊 Confidence Score", f"{data['confidence']:.1%}")
        st.metric("📍 Row Index", data["row_index"])

    with col2:
        st.write("**📝 Proposed Changes:**")
        st.info(data["text"])

    # ==================== BEFORE/AFTER COMPARISON ====================
    st.markdown("---")
    st.subheader("🔄 Changes to Apply")
    
    if ss.changes_preview and len(ss.changes_preview) > 0:
        st.write("**The following changes will be applied:**")
        st.write("")
        
        for change in ss.changes_preview:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**📌 {change['Field']}**")
            with col2:
                st.write(f"**Before:** {change['Before']}")
                st.write(f"**After:** {change['After']}")
            st.markdown("---")
        
        st.caption("💡 These changes will update the Excel file when you click Accept")
    else:
        st.info("ℹ️ No changes detected between current and new values")

    st.warning("⚠️ **Changes have NOT been applied yet.** Click 'Accept' to update the Excel file.")

    st.markdown("### ⚡ Actions")
    col_accept, col_reject, col_cancel = st.columns([1, 1, 2])

    with col_accept:
        if st.button("✅ Accept Changes", type="primary", use_container_width=True):
            row_idx = data["row_index"]

            # Ensure metadata columns exist
            required_cols = ["Last_Updated", "Updated_By", "Change_Reason"]
            for col in required_cols:
                if col not in ss.df.columns:
                    ss.df[col] = ""

            # Apply all changes from changes_preview
            for change in ss.changes_preview:
                field = change['Field']
                new_value = change['After']
                if field in ss.df.columns:
                    ss.df.at[row_idx, field] = new_value

            # Update metadata columns
            ss.df.at[row_idx, "Last_Updated"] = data['timestamp']
            ss.df.at[row_idx, "Updated_By"] = REVIEWER
            ss.df.at[row_idx, "Change_Reason"] = f"{data['category']}: {data['text'][:100]}"

            # SAVE TO ORIGINAL FILE - OVERWRITE WITH NEW VALUES
            if ss.excel_file_path:
                if save_excel_file(ss.df, ss.excel_file_path):
                    st.success("✅ Excel file has been updated and saved with new values!")
                else:
                    st.error("❌ Failed to save Excel file")

            # Add to audit log
            audit_entry = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Reviewer": REVIEWER,
                "Row": data["row_index"],
                "Doctor": f"{data['first_name']} {data['last_name']}",
                "Email": data["email"],
                "Category": data["category"],
                "Action": "Accepted",
                "Changes": data["text"][:100] + "..." if len(data["text"]) > 100 else data["text"]
            }
            ss.audit.append(audit_entry)

            # Clear state
            ss.analyzed_data = None
            ss.match_results = None
            ss.changes_preview = None
            ss.selected_row_idx = None

            st.success("✅ Changes accepted and applied to Excel data!")
            st.info("📥 The Excel file now contains the NEW values. Download to see the updates.")
            time.sleep(2)
            st.rerun()

    with col_reject:
        if st.button("❌ Reject Changes", type="secondary", use_container_width=True):
            audit_entry = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Reviewer": REVIEWER,
                "Row": data["row_index"],
                "Doctor": f"{data['first_name']} {data['last_name']}",
                "Email": data["email"],
                "Category": data["category"],
                "Action": "Rejected",
                "Changes": data["text"][:100] + "..." if len(data["text"]) > 100 else data["text"]
            }
            ss.audit.append(audit_entry)

            st.warning("⚠️ Changes rejected! No updates were applied.")
            time.sleep(1)
            ss.analyzed_data = None
            ss.changes_preview = None
            ss.selected_row_idx = None
            st.rerun()

    with col_cancel:
        if st.button("🔄 Cancel (analyze new text)", use_container_width=True):
            ss.analyzed_data = None
            ss.changes_preview = None
            ss.selected_row_idx = None
            st.rerun()

# ==================== Audit Log ====================
st.markdown("---")
st.header("📊 Audit Log & Download")

if ss.audit:
    audit_df = pd.DataFrame(ss.audit)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Audit Log"):
            ss.audit = []
            st.rerun()

    with col2:
        if ss.df is not None:
            # Download CSV with NEW updated values
            csv = ss.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download as CSV",
                csv,
                "updated_doctors.csv",
                "text/csv",
                use_container_width=True,
                help="Download CSV with all accepted changes applied"
            )

    with col3:
        if ss.df is not None:
            # Download Excel with NEW updated values
            df_to_save = ss.df.copy()
            buffer = BytesIO()

            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_to_save.to_excel(writer, index=False, sheet_name='Sheet1')

            buffer.seek(0)
            st.download_button(
                "💾 Download Excel File",
                data=buffer,
                file_name=ss.excel_file if ss.excel_file else "updated_doctors.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Download Excel file with all accepted changes applied"
            )
else:
    st.info("ℹ️ No audit entries yet. Accept or reject changes to see them here.")

    if ss.df is not None:
        st.subheader("📥 Download Excel File")

        # Download Excel with current state
        df_to_save = ss.df.copy()
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_to_save.to_excel(writer, index=False, sheet_name='Sheet1')

        buffer.seek(0)

        st.download_button(
            "💾 Download Excel File",
            data=buffer,
            file_name=ss.excel_file if ss.excel_file else "updated_doctors.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the current Excel file"
        )
